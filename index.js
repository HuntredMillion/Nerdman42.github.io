const axios = require("axios"); 
var Chart = require('chart.js');
require('chartjs-plugin-crosshair') 

// var tf = require('@tensorflow/tfjs-node');

var xValues = [];
var yValues = [];
var smaValues = [];
var smaSet = [];
var trainNum;
var xMax;
var xMin;
var yMax;
var yMin;
var cancelThrown = false;
var modelStoppedThrown = false;
var smaDisp = [];

/////////MAIN FUNCTIONS
       
function search(query){
  if(xValues.length != 0){ //destroys graph when doing new query, resets HTML
    xValues = [];
    yValues = [];
    aaChart.destroy();
    const bill = document.getElementById('load');
    if(bill.style.visibility === "visible"){ //using as flag to see if model is in training process
      cancelThrown = true; //kills model training
      bill.style.visibility === "hidden"
    }
    const epochData = document.getElementById('LSTM_data');
    setTimeout(() => { epochData.innerHTML = ''; }, 4000); //set delay for clearing page
  }
  query.preventDefault();
  query.target.elements["submit"].disabled = true;
  
  
  let tick = query.target.elements["search"].value; //gets data from HTML
  let winSize = parseInt(query.target.elements["window_size"].value); 
  let distance = parseInt(query.target.elements["distance"].value);
  let epochNum = parseInt(query.target.elements["epochNum"].value);
  let freq = query.target.elements["freq"].value;
  let dataSplit = parseInt(query.target.elements["dataSplit"].value);
  let hidLayers = parseInt(query.target.elements["hidLayers"].value);
  let LearningRate = parseFloat(query.target.elements["LearningRate"].value);
  initData(tick, winSize, freq, dataSplit);

  setTimeout(() => {  query.target.elements["submit"].disabled = false; }, 5000);
  setTimeout(() => {  mein(winSize, distance, epochNum, dataSplit, hidLayers, LearningRate, freq, tick); }, 5000); //set delay for axios to grab data
}


async function mein(winSize, distance, epochNum, dataSplit, hidLayers, LearningRate, freq, tick){
  showChart(tick); //displays ticker data
  let mod = await trainModel(winSize, dataSplit, hidLayers, LearningRate, epochNum, distance); 

  if(!modelStoppedThrown){
    var trainX = smaSet.slice(winSize, trainNum-distance); //creates data sets for validation
    var validX = smaSet.slice(trainNum- distance);
    let trainY = validateModel(mod, trainX, winSize, distance); //creates validation
    let validY = validateModel(mod, validX, winSize, 0);
    updateChart(trainY, winSize, false) //updates chart
    updateChart(validY, winSize, true)
    predictData(freq, distance);
  }
  modelStoppedThrown = false;
}

something.addEventListener("submit", search, false)












///DATA INITIALIZATION

function initData(ticker, n_steps, freq, trainSize){

  axios.request({//fetches historical stock data
  url: 'https://yh-finance.p.rapidapi.com/stock/v2/get-chart',
  params: {
    interval: freq,
    symbol: ticker,
    range: '10y',
    region: 'US',
  },
  headers: {
    'X-RapidAPI-Key': '8798a229ffmshad68daf26d2dc80p18d220jsnb927abdecbe2', 
    'X-RapidAPI-Host': 'yh-finance.p.rapidapi.com'
  }}).then(function(response) {
    var stock_data = response.data.chart.result['0'];
    console.log(response.data);
    stock_data.timestamp.forEach(function (item, index) { //converts timestamp data from seconds since epoch to US date format
      const unixTime = item;
      const date = new Date(unixTime*1000);
      xValues.push(date.toLocaleDateString("en-US")); //pushes dates to an array --> used for labels in chart
    });
    
    stock_data.indicators.quote[0].close.forEach(function (item, index) {
      yValues.push(item); //pushes stock values to array
    });
    if(freq==="1wk"){ //if data collected weekly..
      dateCorrection(); //corrects issue of last data point not being in weekly
    }
    var smaArr = get_sma(stock_data.indicators.quote[0].close, n_steps);
    smaValues = smaArr[0]; //creates array of SMA data
    smaSet = smaArr[1];
    trainNum = Math.floor(trainSize / 100 * smaSet.length); //dividing num between training set and validation set
    smaDisp = smaValues.slice(0, Math.floor(trainSize / 100 * smaValues.length));
    
  }).catch(function(error) {
    console.error(error);
  });
}

function dateCorrection(){ 
  var lastDate = xValues[xValues.length-1];
  var nextLast = xValues[xValues.length-2]; //gets last two dates
  var lDate = new Date(lastDate);
  var nDate = new Date(nextLast); //converts to Date objects
  const weekMs = 60*60*24*7*1000;
  var dateBack = lDate - weekMs; 
  var diff = dateBack - nDate; //checks if last two dates are 7 days apart
  if(!diff==0){
    xValues.pop() //if not, the data is excluded
    yValues.pop()
  }
}

function get_sma(data, window_size){ //creates SMA, or Simple Moving Average, along with SMA datasets
  let avg_idx = [];
  let avg_set = [];
  for(let i=0;i<window_size;i++){ //push null data so the dataset aligns with the labels already on the chart
    avg_idx.push(null);
    avg_set.push(null);
  }
  for(let i = window_size; i< data.length; i++){ //goesh through each sma value
    let avg = 0.00, t = i - window_size;
    let temp_set =[];
    for(let k = t; k < i && k <data.length; k++){//goes through each value for the sma
      avg += data[k] / window_size;
      temp_set.push(data[k]);
    }
    avg_idx.push(avg); //array of n-size averages
    avg_set.push(temp_set); //array of n-size arrays for each average
  }
  return [avg_idx, avg_set];
}














//////DATA MODELING
async function trainModel(n_size, trainSize, hidLayers, LearningRate, epochNum, dist){ //the workhorse--this creates the model

  try{
    const batch_size = 32;
    
    var X = smaSet.slice(n_size, trainNum); //creates training set
    var Y = smaValues.slice(n_size, Math.floor(trainSize / 100 * smaValues.length));
    for(let i=0; i<dist;i++){ //creates prediction offset
      X.pop();
      Y.shift();
    }
    var Xt = tf.tensor2d(X, [X.length, X[0].length]);//training set to tensor
    var Yt = tf.tensor2d(Y, [Y.length, 1])


    var inputs
    var outputs
    [inputs, xMax, xMin] = normalizeTensorFit(Xt); //normalizes training data
    [outputs, yMax, yMin] = normalizeTensorFit(Yt);
    

    const model = tf.sequential();

    model.add(tf.layers.dense({units: 64, inputShape: [n_size]})); //these two layers set the shape for the model input and allow transition
    model.add(tf.layers.reshape({targetShape: [16,4] }));//to the LSTM layers

    
    let lstm_cells = [];
    for (let index = 0; index < hidLayers; index++) {  //creates hidden layers
        lstm_cells.push(tf.layers.lstmCell({units: 16}));
    }

    model.add(tf.layers.rnn({ //adds hidden layers to model
      cell: lstm_cells,
      inputShape: [16,4],
      
    }));

    model.add(tf.layers.dense({units: 1, inputShape: [16]})); //a sort of culminating layer, the fully-connected 'dense' layer 
    //is connected to every neuron of of preceding LSTM layer

    model.compile({ //compiles layers
      optimizer: tf.train.adam(LearningRate),
      loss: 'meanSquaredError'
    });




    var load = document.getElementById("load");//sets up HTML logging
    load.style.visibility = "visible";

    const hist = await model.fit(inputs, outputs,   //uploads datasets and fits the model
      { batchSize: batch_size, epochs: epochNum, callbacks: {
        onEpochEnd: async (epoch, log) => {
          console.log(log)
          var logStr = log.loss;
          var epochInt = parseInt(epoch)+1;
          logEpoch(epochInt,logStr); //logs data from each epoch
          if(cancelThrown){
            modelStoppedThrown = true;
            cancelThrown=false;
            model.stopTraining = true;
          }
        }
      }
    });

    load.style.visibility = "hidden";
    return model;

  } catch(error){
    console.log("error")
    return null;
  }  
}


function logEpoch(epoch,log){ //logs epoch data
  var element = document.getElementById("LSTM_data");
  var para = document.createElement("p");
  para.style.cssText = 'font-size:16px';
  var node = document.createTextNode("     epoch "+ epoch +": "+log);
  para.appendChild(node);
  
  element.appendChild(para);
}

function normalizeTensorFit(tens){ //creates tensor of data normalized between 0 and 1
  const maxval = tf.max(tens);
  const minval = tf.min(tens);
  
  const normalizedTensor = tf.sub(tens, minval).div(tf.sub(maxval,minval));
  return [normalizedTensor, maxval, minval];
}



function validateModel(model, dataX, winSize, dist){ //uses model to predict values based on n-length datasets
  let dataTen = tf.tensor2d(dataX, [dataX.length, winSize])
  let dataNorm = tf.sub(dataTen, xMin).div(tf.sub(xMax, xMin));//normalizes data
  let dataOut = model.predict(dataNorm);
  let outNorm = tf.add(dataOut.mul(tf.sub(yMax, yMin)), yMin); //un-normalizes data
  let dataY = Array.from(outNorm.dataSync());
  for(let i=0; i<dist;i++){ //creates prediction offset
    dataY.unshift(null);
  }
  return dataY; //returs array of predicted values
}


function predictData(freq, dist){ //horribly named, but too lazy to change
  var lastDate = xValues[xValues.length-1];
  var lDate = new Date(lastDate);
  var dateMs = lDate -0;
  let weekMs = 60*60*24*7*1000;
  let dayMs = weekMs/7;
  

  if(freq==="1wk"){
    for(let i=0;i<dist;i++){
      
      let newDate = new Date(dateMs + weekMs);
      aaChart.data.labels.push(newDate.toLocaleDateString("en-US"));
      dateMs += weekMs;
    }
  }else{
    for(let i=0;i<dist;i++){
      let newDate = new Date(dateMs + dayMs);
      aaChart.data.labels.push(newDate.toLocaleDateString("en-US"));
      dateMs = newDate;
    }
  }
  aaChart.update();
}



//CHART FUNCTIONS
////////////////
///////////////
//////////////
/////////////
////////////
///////////
//////////
/////////
////////
///////
//////
/////
////
///
//
function updateChart(newData, n_size, valid){ //add modeled data to chart
  if (valid){
    for(let i=0; i<trainNum; i++){
      newData.unshift(null); //shifts data to align with chart labels
    }
    var newDataset = {
      label: 'Predicted Result',
      data: newData,
      borderColor: "rgb(0,0,0)",
      backgroundColor: "rgb(0,0,0)",
      fill: false,
      lineTension: 0,
      borderWidth: 2,
      spanGaps: true,
      pointRadius: 0,
    }

  }else {
    for(let i=0; i<n_size; i++){
      newData.unshift(null);
    }
    var newDataset = {
      label: 'Training Result',
      data: newData,
      borderColor: "rgb(0,240,0)",
      backgroundColor: "rgb(0,240,0)",
      fill: false,
      lineTension: 0,
      borderWidth: 2,
      spanGaps: true,
      pointRadius: 0,
    }

  }

  // Adds newly created dataset to list of `data`
  aaChart.data.datasets.push(newDataset);

  //Updates the chart 
  aaChart.update();
}



function showChart(tick){ //initializes chart
  var ctx = document.getElementById("myyChart");
  aaChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: xValues,
      datasets: [
        { 
          label: 'Stock',
          data: yValues,
          borderColor: "rgb(0,0,255)",
          backgroundColor: "rgb(0,0,255)",
          fill: false,
          lineTension: 0,
          borderWidth: 1,
          spanGaps: true,
          pointRadius: 0,
          
        },
        {
          label: 'SMA',
          data: smaDisp,
          borderColor: "rgb(255,0,0)",
          backgroundColor: "rgb(255,0,0)",
          fill: false,
          lineTension: 0,
          borderWidth: 1,
          spanGaps: true,
          pointRadius: 0,
          
      }]
    },
    options: {
      plugins: {
        title: {
          display: true,
          text: tick + ' Stock Prices and Prediction'
        },
        tooltip: {
        },
        crosshair: {
          line: {
            color: '#F66',  // crosshair line color
            width: 1        // crosshair line width
          },
          sync: {
            enabled: true,            // enable trace line syncing with other charts
            group: 1,                 // chart group
            suppressTooltips: false   // suppress tooltips when showing a synced tracer
          },
          zoom: {
            enabled: true,                                      // enable zooming
            zoomboxBackgroundColor: 'rgba(66,133,244,0.2)',     // background color of zoom box 
            zoomboxBorderColor: '#48F',                         // border color of zoom box
            zoomButtonText: 'Reset Zoom',                       // reset zoom button text
            zoomButtonClass: 'reset-zoom',                      // reset zoom button class
          },
          callbacks: {
            beforeZoom: () => function(start, end) {                  // called before zoom, return false to prevent zoom
              return true;
            },
            afterZoom: () => function(start, end) {                   // called after zoom
            }
          }
        }
      },
      interaction: {intersect: false, mode:'nearest', axis: 'x'}, //how the cursor interacts with the datapoints
      legend: {display: false},
      scales: {
        
        x: {
          ticks: {
              autoSkip: true,
              maxTicksLimit: 10
          }
        }
      }
    }
  });
}