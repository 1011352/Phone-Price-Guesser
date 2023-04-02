import { createChart, updateChart } from "./scatterplot.js";

const nn = ml5.neuralNetwork({ task: "regression", debug: true });
let button = document.getElementById("btn");
let result = document.getElementById("result");

function loadData() {
    Papa.parse("./data/mobilephones.csv", {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: (results) => plotData(results.data),
    });
}
//
// demo data
//

function plotData(data) {
    data.sort(() => Math.random() - 0.5);
    let trainData = data.slice(0, Math.floor(data.length * 0.8))
    let testData = data.slice(Math.floor(data.length * 0.8) + 1)

    // een voor een de data toevoegen aan het neural network
    for (let phone of trainData) {
        nn.addData({ sale: phone.sale, weight: phone.weight, resolution: phone.resolution }, { price: phone.price })
}

    nn.normalizeData();

    let chartdata = data.map((phone) => ({
        x: phone.sale,
        x: phone.weight,
        x: phone.resolution,
        y: phone.price,
    }));

    createChart(chartdata, "sale", "price");
    startTraining();

    button.addEventListener("click", () => {
        let sale = parseInt(document.getElementById("sale").value);
        let weight = parseInt(document.getElementById("weight").value);
        let resolution = parseInt(document.getElementById("resolution").value);
        makePrediction(sale, weight, resolution);
    });
}
// const data = [
//         { horsepower: 130, mpg: 18 },
//         { horsepower: 165, mpg: 15 },
//         { horsepower: 225, mpg: 14 },
//         { horsepower: 97, mpg: 18 },
//         { horsepower: 88, mpg: 27 },
//         { horsepower: 193, mpg: 9 },
//         { horsepower: 80, mpg: 25 },
// ]

function startTraining() {
    nn.train({ epochs: 15 }, () => finishedTraining());
}

async function makePrediction(sale, weight, resolution) {
    const results = await nn.predict({ sale: sale , weight: weight, resolution: resolution });
    result.innerHTML = `Geschat verbruik: ${results[0].price}`;
}

async function finishedTraining() {
    // let predictions = []
    // for (let sl = 10; sl < 1000; sl += 20) {
    //     const pred = await nn.predict({sale: sl})
    //     predictions.push({x: sl,  y: pred[0].price})
    // }
    // updateChart("Predictions", predictions)
}

loadData();
