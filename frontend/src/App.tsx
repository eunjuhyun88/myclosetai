
import React, { useState, useRef } from 'react';
import { Upload, Camera, Zap, Activity, CheckCircle, AlertCircle } from 'lucide-react';
import './App.css';

interface TryOnResult {
  success: boolean;
  fitted_image: string;
  processing_time: number;
  confidence: number;
  fit_score: number;
  recommendations: string[];
}

function App() {
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [clothingImage, setClothingImage] = useState<File | null>(null);
  const [height, setHeight] = useState<number>(170);
  const [weight, setWeight] = useState<number>(65);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<TryOnResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const personInputRef = useRef<HTMLInputElement>(null);
  const clothingInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (file: File, type: 'person' | 'clothing') => {
    if (type === 'person') {
      setPersonImage(file);
    } else {
      setClothingImage(file);
    }
    setError(null);
  };

  const processVirtualTryOn = async () => {
    if (!personImage || !clothingImage) {
      setError('Please upload both images');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('clothing_image', clothingImage);
      formData.append('height', height.toString());
      formData.append('weight', weight.toString());
      formData.append('model_type', 'ootd');
      formData.append('category', 'upper_body');

      const response = await fetch('http://localhost:8000/api/virtual-tryon', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Processing failed');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-orange-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-orange-500 rounded-lg flex items-center justify-center">
                <Camera className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-2xl font-bold text-gray-900">MyCloset AI</h1>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span className="text-sm text-gray-600">Server Online</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">AI Virtual Try-On</h2>
          <p className="text-gray-600 max-width-2xl mx-auto">
            Experience the future of fashion with our AI-powered virtual fitting room. 
            Upload your photo and see how clothes look on you instantly.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Upload Section */}
          <div className="space-y-6">
            {/* Person Image Upload */}
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Your Photo</h3>
              {personImage ? (
                <div className="relative">
                  <img
                    src={URL.createObjectURL(personImage)}
                    alt="Person"
                    className="w-full h-64 object-cover rounded-lg"
                  />
                  <button
                    onClick={() => personInputRef.current?.click()}
                    className="absolute top-2 right-2 bg-white rounded-full p-2 shadow-lg hover:bg-gray-50"
                  >
                    <Upload className="w-4 h-4 text-gray-600" />
                  </button>
                </div>
              ) : (
                <div
                  onClick={() => personInputRef.current?.click()}
                  className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center cursor-pointer hover:border-orange-400 transition-colors"
                >
                  <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-gray-600">Upload your photo</p>
                  <p className="text-sm text-gray-400">PNG, JPG up to 10MB</p>
                </div>
              )}
              <input
                ref={personInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'person')}
                className="hidden"
              />
            </div>

            {/* Clothing Image Upload */}
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Clothing Item</h3>
              {clothingImage ? (
                <div className="relative">
                  <img
                    src={URL.createObjectURL(clothingImage)}
                    alt="Clothing"
                    className="w-full h-64 object-cover rounded-lg"
                  />
                  <button
                    onClick={() => clothingInputRef.current?.click()}
                    className="absolute top-2 right-2 bg-white rounded-full p-2 shadow-lg hover:bg-gray-50"
                  >
                    <Upload className="w-4 h-4 text-gray-600" />
                  </button>
                </div>
              ) : (
                <div
                  onClick={() => clothingInputRef.current?.click()}
                  className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center cursor-pointer hover:border-orange-400 transition-colors"
                >
                  <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-gray-600">Upload clothing item</p>
                  <p className="text-sm text-gray-400">PNG, JPG up to 10MB</p>
                </div>
              )}
              <input
                ref={clothingInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'clothing')}
                className="hidden"
              />
            </div>
          </div>

          {/* Controls and Results */}
          <div className="space-y-6">
            {/* Measurements */}
            <div className="bg-white rounded-xl shadow-sm border p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Body Measurements</h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Height (cm)</label>
                  <input
                    type="number"
                    value={height}
                    onChange={(e) => setHeight(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                    min="140"
                    max="220"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Weight (kg)</label>
                  <input
                    type="number"
                    value={weight}
                    onChange={(e) => setWeight(Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
                    min="30"
                    max="150"
                  />
                </div>
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={processVirtualTryOn}
              disabled={!personImage || !clothingImage || isProcessing}
              className="w-full bg-orange-500 text-white py-4 px-6 rounded-xl font-semibold disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-orange-600 transition-colors flex items-center justify-center space-x-2"
            >
              {isProcessing ? (
                <>
                  <Activity className="w-5 h-5 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>Generate Try-On</span>
                </>
              )}
            </button>

            {/* Error Display */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
            )}

            {/* Results Display */}
            {result && (
              <div className="bg-white rounded-xl shadow-sm border p-6">
                <div className="flex items-center mb-4">
                  <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                  <h3 className="text-lg font-semibold text-gray-900">Try-On Complete!</h3>
                </div>
                
                <img
                  src={`data:image/jpeg;base64,${result.fitted_image}`}
                  alt="Virtual Try-On Result"
                  className="w-full rounded-lg mb-4"
                />
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="font-medium">Fit Score:</span>
                    <span className="ml-2 text-green-600">{Math.round(result.fit_score * 100)}%</span>
                  </div>
                  <div>
                    <span className="font-medium">Confidence:</span>
                    <span className="ml-2 text-blue-600">{Math.round(result.confidence * 100)}%</span>
                  </div>
                </div>

                {result.recommendations && result.recommendations.length > 0 && (
                  <div className="mt-4">
                    <h4 className="font-medium text-gray-900 mb-2">AI Recommendations:</h4>
                    <ul className="space-y-1">
                      {result.recommendations.map((rec, index) => (
                        <li key={index} className="text-sm text-gray-600">• {rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
