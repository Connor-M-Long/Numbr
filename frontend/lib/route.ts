"use server";

export const get = async () => {
    const data = await fetch("http://localhost:8000/getPredictions");
    return data.json()
};