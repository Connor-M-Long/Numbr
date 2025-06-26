import { get } from "../../lib/route";
import path from 'path';
import fs from 'fs';
import axios from 'axios';

export default async function Home() {
  const apicall = await get();

  const imagePath = path.join(process.cwd(), 'public', 'images', 'number.png');

  return (
    <>
    <div>
      <h1>Numbr</h1>
    </div>

    <div>
      <p>Prediction: {apicall.Prediction}</p>
      <img src={apicall.img}/>
    </div>
    
    
    </>
  );
}
