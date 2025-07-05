"use server";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

export const get = async () => {
    const data = await fetch(BACKEND_URL + '/getPredictions');
    return data.json()
};

export const train = async () => {
    const data = await fetch(BACKEND_URL + "/trainModel");
    return data.json()
};