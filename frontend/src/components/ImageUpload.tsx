import React, { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

interface ImageUploadProps {
    onImageUpload: (file: File) => void;
    isLoading: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onImageUpload, isLoading }) => {
    const onDrop = useCallback((acceptedFiles: File[]) => {
        if (acceptedFiles && acceptedFiles.length > 0) {
            onImageUpload(acceptedFiles[0])
        }
    }, [onImageUpload])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.png', '.jpg']
        },
        multiple: false,
        disabled: isLoading
    })

    return (
        <div
            {...getRootProps()}
            className={`
                relative overflow-hidden group rounded-2xl border-2 border-dashed transition-all duration-300 ease-in-out cursor-pointer p-10
                ${isDragActive
                    ? 'border-blue-500 bg-blue-50 scale-[1.02] ring-4 ring-blue-100'
                    : 'border-slate-300 hover:border-blue-400 hover:bg-slate-50'
                }
                ${isLoading ? 'opacity-50 cursor-not-allowed pointer-events-none' : ''}
            `}
        >
            <input {...getInputProps()} />

            <div className="flex flex-col items-center justify-center text-center space-y-4">
                <div className={`
                    p-4 rounded-full bg-blue-100 text-blue-600 transition-transform duration-300
                    ${isDragActive ? 'scale-110 rotate-12' : 'group-hover:scale-110'}
                `}>
                    <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                </div>

                <div className="space-y-1">
                    <p className="text-lg font-semibold text-slate-700">
                        {isDragActive ? "Drop it like it's hot!" : "Upload Retinal Scan"}
                    </p>
                    <p className="text-sm text-slate-500">
                        Drag & drop or <span className="text-blue-600 font-medium hover:underline">browse</span>
                    </p>
                </div>

                <div className="flex items-center gap-2 text-xs text-slate-400 uppercase tracking-wide font-medium">
                    <span className="bg-slate-100 px-2 py-1 rounded">JPG</span>
                    <span className="bg-slate-100 px-2 py-1 rounded">PNG</span>
                    <span>Max 10MB</span>
                </div>
            </div>
        </div>
    )
}

export default ImageUpload
