import React, { useState } from 'react'
import axios from 'axios'
import ImageUpload from './components/ImageUpload'
import ResultDisplay from './components/ResultDisplay'

interface PredictionResult {
    filename: string;
    prediction_class: number;
    prediction_label: string;
    confidence: number;
}

function App() {
    const [isLoading, setIsLoading] = useState(false)
    const [result, setResult] = useState<PredictionResult | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [previewUrl, setPreviewUrl] = useState<string | null>(null)

    const handleImageUpload = async (file: File) => {
        setIsLoading(true)
        setError(null)
        setResult(null)
        setPreviewUrl(URL.createObjectURL(file))

        const formData = new FormData()
        formData.append('file', file)

        try {
            const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
            const response = await axios.post(`${apiUrl}/predict`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            })
            setResult(response.data)
        } catch (err) {
            console.error(err)
            setError("Failed to analyze image. Please try again.")
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="min-h-screen flex flex-col items-center justify-center p-4 sm:p-6 lg:p-8">
            <div className="w-full max-w-4xl bg-white/60 backdrop-blur-xl rounded-3xl shadow-2xl overflow-hidden border border-white/50 animate-fade-in-up">

                {/* Header */}
                <div className="bg-gradient-to-r from-blue-600 to-indigo-700 py-10 px-8 text-center relative overflow-hidden">
                    <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
                    <h1 className="text-4xl sm:text-5xl font-extrabold text-white tracking-tight drop-shadow-sm mb-2">
                        Diabetic Retinopathy Detection
                    </h1>
                    <p className="text-blue-100 text-lg font-light max-w-2xl mx-auto">
                        Advanced AI-powered analysis for early detection and grading using deep learning.
                    </p>
                </div>

                <div className="p-8 sm:p-12 space-y-10">
                    {/* Upload Section */}
                    <div className="space-y-6">
                        <ImageUpload onImageUpload={handleImageUpload} isLoading={isLoading} />

                        {previewUrl && (
                            <div className="relative group rounded-2xl overflow-hidden shadow-md max-w-sm mx-auto border-4 border-white">
                                <img
                                    src={previewUrl}
                                    alt="Preview"
                                    className="w-full h-64 object-cover transform transition-transform duration-500 group-hover:scale-105"
                                />
                                <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors"></div>
                            </div>
                        )}

                        {isLoading && (
                            <div className="flex flex-col items-center justify-center py-6 space-y-3">
                                <div className="relative w-16 h-16">
                                    <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-200 rounded-full opacity-50"></div>
                                    <div className="absolute top-0 left-0 w-full h-full border-4 border-blue-600 rounded-full animate-spin border-t-transparent"></div>
                                </div>
                                <p className="text-blue-700 font-medium animate-pulse">Analyzing retinal structures...</p>
                            </div>
                        )}

                        {error && (
                            <div className="p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-r-lg shadow-sm">
                                <p className="font-semibold">Error</p>
                                <p>{error}</p>
                            </div>
                        )}
                    </div>

                    {/* Result Section */}
                    {result && (
                        <div className="animate-fade-in">
                            <ResultDisplay
                                predictionClass={result.prediction_class}
                                predictionLabel={result.prediction_label}
                                confidence={result.confidence}
                            />
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="bg-gray-50 px-8 py-4 border-t border-gray-100 text-center text-gray-400 text-sm">
                    Powered by <span className="font-semibold text-gray-500">TensorFlow</span> & <span className="font-semibold text-gray-500">FastAPI</span>
                </div>
            </div>
        </div>
    )
}

export default App
