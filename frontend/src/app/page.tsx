"use client";
import { get, train } from "../../lib/route";
import React, {useEffect, useState} from 'react';

export default function Home() {
  const [apicall_getData, pred] = useState({
            "Prediction": "",
            "Label": "",
            "img": ""
        });

  const [apicall_trainData, traindata] = useState({
            "W1": "",
            "W2": "",
            "B1": "",
            "B2": ""
        });
  
  const [refreshKey, setRefreshKey] = React.useState(Date.now());
        
  const refreshImage = () => {
    setRefreshKey(Date.now());
  };

  const predict = async () => {
    const data = await get();
    pred(data);
    refreshImage();
  }

  const training = async () => {
    const data = await train(); 
    traindata(data);
  }

  const [backend_URL, set_b_URL] = useState<string | null>(null);

  useEffect(() => {
    const fetchURL = async () => {
        const b_URL = (`${process.env.NEXT_PUBLIC_BACKEND_URL}/static/images/Num.png`);
        set_b_URL(b_URL);
    };

    fetchURL();
  }, []);

  useEffect(() => {
    predict();
  }, []);

  return (
    <>
    <div className="text-white text-center">
      <h1 className="text-6xl">Numbr</h1>
      <p className="m-3">An image classification system, utilising a neural network that I have built</p>
      <p>This is a showcase of how this network can be utilised</p>
    </div>

    <div className="text-white text-center border border-white-500 w-200 mx-auto mt-10">
      <img className="w-150 h-1/4 mx-auto m-10" src={backend_URL ? `${backend_URL}?cache_bust=${refreshKey}` : ""}/>
      <div className="m-3">Prediction: {apicall_getData.Prediction}</div>
    </div>

    <div className="text-white text-center">
      <button className="border-1 rounded-md w-24 m-5" onClick={predict}>Predict</button>
      <button className="border-1 rounded-md w-24 m-5" onClick={training}>Train</button>
    </div>
    
    <div id="trainingData" className="flex justify-center">
      <div className="border border-white-500 w-[1000px] h-[500px] overflow-y-auto mt-10 mb-10">
      <table className="table-fixed border-collapse overflow-y-contain border border-white-500">
        <thead>
          <tr>
            <th className="border px-4 py-2 w-[250px]">Weight 1</th>
            <th className="border px-4 py-2 w-[250px]">Weight 2</th>
            <th className="border px-4 py-2 w-[250px]">Bias 1</th>
            <th className="border px-4 py-2 w-[250px]">Bias 2</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="border px-4 py-2 align-top">{apicall_trainData.W1}</td>
            <td className="border px-4 py-2 align-top">{apicall_trainData.W2}</td>
            <td className="border px-4 py-2 align-top">{apicall_trainData.B1}</td>
            <td className="border px-4 py-2 align-top">{apicall_trainData.B2}</td>
          </tr>
        </tbody>
      </table>
      </div>
    </div>
    </>
  );
}
