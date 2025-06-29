"use client";
import { get, train } from "../../lib/route";
import React, {useState} from 'react';
import path from 'path';
import fs from 'fs';
import axios from 'axios';

export default function Home() {
  const [apicall_getData, pred] = useState({
            "Prediction": "",
            "Label": "",
        });

  const [apicall_trainData, traindata] = useState({
            "W1": "",
            "W2": "",
            "B1": "",
            "B2": ""
        });
  
  const predict = async () => {
    const data = await get();
    pred(data);
  }

  const training = async () => {
    const data = await train(); 
    traindata(data);
  }

  return (
    <>
    <div className="bg-red-500">
      <h1>Numbr</h1>
      <p>An image classification system, utilising a neural network that I have built</p>
      <p>This is a showcase of how this network can be utilised</p>
    </div>

    <div>
      <div>Prediction: {apicall_getData.Prediction}</div>
    </div>

    <div className="flex flex-row">
      <div>Weight 1: {apicall_trainData.W1}</div>
      <div>Weight 2: {apicall_trainData.W2}</div>
      <div>Bias 1: {apicall_trainData.B1}</div>
      <div>Bias 2: {apicall_trainData.B2}</div>
    </div>

    <div>
      <button className="border-1 rounded-md w-24" onClick={predict}>Predict</button>
      <button className="border-1 rounded-md w-24" onClick={training}>Train</button>
    </div>
    
    
    </>
  );
}
