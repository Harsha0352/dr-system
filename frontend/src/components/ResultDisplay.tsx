import React from 'react'

interface ResultDisplayProps {
    predictionClass: number;
    predictionLabel: string;
    confidence: number;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ predictionClass, predictionLabel, confidence }) => {
    const getSeverityConfig = (severity: number) => {
        switch (severity) {
            case 0: return { color: 'bg-green-500', text: 'text-green-700', bg: 'bg-green-50', label: 'Healthy' };
            case 1: return { color: 'bg-yellow-400', text: 'text-yellow-700', bg: 'bg-yellow-50', label: 'Mild' };
            case 2: return { color: 'bg-orange-400', text: 'text-orange-700', bg: 'bg-orange-50', label: 'Moderate' };
            case 3: return { color: 'bg-red-500', text: 'text-red-700', bg: 'bg-red-50', label: 'Severe' };
            case 4: return { color: 'bg-red-700', text: 'text-red-900', bg: 'bg-red-100', label: 'Proliferative' };
            default: return { color: 'bg-gray-400', text: 'text-gray-700', bg: 'bg-gray-50', label: 'Unknown' };
        }
    }

    const config = getSeverityConfig(predictionClass);
    const confidencePercent = (confidence * 100).toFixed(1);

    return (
        <div className="bg-white rounded-xl overflow-hidden border border-slate-200 shadow-lg">
            <div className={`${config.bg} p-6 border-b border-slate-100`}>
                <h3 className="text-lg font-bold text-slate-700 uppercase tracking-wider mb-1">Diagnosis Result</h3>
                <div className="flex items-center gap-3">
                    <span className={`text-3xl font-extrabold ${config.text}`}>
                        {predictionLabel}
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide border ${config.text} border-current opacity-70`}>
                        Grade {predictionClass}
                    </span>
                </div>
            </div>

            <div className="p-6 space-y-6">
                <div>
                    <div className="flex justify-between items-end mb-2">
                        <span className="text-sm font-semibold text-slate-600">AI Confidence Score</span>
                        <span className="text-2xl font-bold text-blue-600">{confidencePercent}%</span>
                    </div>
                    <div className="h-4 bg-slate-100 rounded-full overflow-hidden shadow-inner">
                        <div
                            className={`h-full ${config.color} transition-all duration-1000 ease-out rounded-full relative`}
                            style={{ width: `${confidencePercent}%` }}
                        >
                            <div className="absolute inset-0 bg-white/20 animate-[shimmer_2s_infinite]"></div>
                        </div>
                    </div>
                    <p className="text-xs text-slate-400 mt-2 text-right">Based on ResNet50 model analysis</p>
                </div>
            </div>
        </div>
    )
}

export default ResultDisplay
