import React, { useState, useRef } from 'react';

interface UserMeasurements {
  height: number;
  weight: number;
}

interface TryOnResult {
  success: boolean;
  fitted_image: string;
  processing_time: number;
  confidence: number;
  measurements: {
    chest: number;
    waist: number;
    hip: number;
    bmi: number;
  };
  clothing_analysis: {
    category: string;
    style: string;
    dominant_color: number[];
  };
  fit_score: number;
  recommendations: string[];
}

const App: React.FC = () => {
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [clothingImage, setClothingImage] = useState<File | null>(null);
  const [measurements, setMeasurements] = useState<UserMeasurements>({
    height: 170,
    weight: 65
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<TryOnResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const personImageRef = useRef<HTMLInputElement>(null);
  const clothingImageRef = useRef<HTMLInputElement>(null);

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
      setError('Please upload both photos to continue');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('clothing_image', clothingImage);
      formData.append('height', measurements.height.toString());
      formData.append('weight', measurements.weight.toString());

      const response = await fetch('http://localhost:8000/api/virtual-tryon', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        setResult(data);
      } else {
        throw new Error(data.error || 'An error occurred during processing');
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during processing');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f9fafb', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto", "Oxygen", "Ubuntu", "Cantarell", "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif' }}>
      {/* Header */}
      <header style={{ backgroundColor: '#ffffff', borderBottom: '1px solid #e5e7eb' }}>
        <div style={{ maxWidth: '80rem', margin: '0 auto', padding: '0 1rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', height: '4rem' }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{ flexShrink: 0 }}>
                <div style={{ 
                  width: '2rem', 
                  height: '2rem', 
                  backgroundColor: '#000000', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center' 
                }}>
                  <svg style={{ width: '1.25rem', height: '1.25rem', color: '#ffffff' }} fill="currentColor" viewBox="0 0 24 24">
                    <path d="M12 2L2 7v10c0 5.55 3.84 10 9 10s9-4.45 9-10V7l-10-5z"/>
                  </svg>
                </div>
              </div>
              <div style={{ marginLeft: '0.75rem' }}>
                <h1 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827', margin: 0 }}>MyCloset AI</h1>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <div style={{ height: '0.5rem', width: '0.5rem', backgroundColor: '#4ade80', borderRadius: '50%' }}></div>
              <span style={{ fontSize: '0.875rem', color: '#4b5563' }}>Server Online</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: '80rem', margin: '0 auto', padding: '2rem 1rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <h2 style={{ fontSize: '1.875rem', fontWeight: '700', color: '#111827', marginBottom: '0.5rem' }}>AI Virtual Try-On</h2>
          <p style={{ color: '#4b5563', maxWidth: '42rem', margin: '0 auto' }}>
            Upload your photo and any clothing item to see how it looks on you. 
            Our AI creates realistic fitting previews in seconds.
          </p>
        </div>

        <div style={{ maxWidth: '56rem', margin: '0 auto' }}>
          {/* Upload Section */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: window.innerWidth > 768 ? 'repeat(2, 1fr)' : '1fr', 
            gap: '1.5rem', 
            marginBottom: '2rem' 
          }}>
            {/* Person Upload */}
            <div style={{ 
              backgroundColor: '#ffffff', 
              borderRadius: '0.75rem', 
              border: '1px solid #e5e7eb', 
              padding: '1.5rem' 
            }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '500', color: '#111827', marginBottom: '1rem' }}>Your Photo</h3>
              {personImage ? (
                <div style={{ position: 'relative' }}>
                  <img
                    src={URL.createObjectURL(personImage)}
                    alt="Person"
                    style={{ 
                      width: '100%', 
                      height: '16rem', 
                      objectFit: 'cover', 
                      borderRadius: '0.5rem' 
                    }}
                  />
                  <button
                    onClick={() => personImageRef.current?.click()}
                    style={{ 
                      position: 'absolute', 
                      top: '0.5rem', 
                      right: '0.5rem', 
                      backgroundColor: '#ffffff', 
                      borderRadius: '50%', 
                      padding: '0.5rem', 
                      boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', 
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    <svg style={{ width: '1rem', height: '1rem', color: '#4b5563' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </button>
                  <div style={{ 
                    position: 'absolute', 
                    bottom: '0.5rem', 
                    left: '0.5rem', 
                    backgroundColor: 'rgba(0,0,0,0.5)', 
                    color: '#ffffff', 
                    fontSize: '0.75rem', 
                    padding: '0.25rem 0.5rem', 
                    borderRadius: '0.25rem' 
                  }}>
                    {personImage.name}
                  </div>
                </div>
              ) : (
                <div 
                  onClick={() => personImageRef.current?.click()}
                  style={{ 
                    border: '2px dashed #d1d5db', 
                    borderRadius: '0.5rem', 
                    padding: '3rem', 
                    textAlign: 'center', 
                    cursor: 'pointer',
                    transition: 'border-color 0.2s'
                  }}
                  onMouseEnter={(e) => (e.target as HTMLElement).style.borderColor = '#9ca3af'}
                  onMouseLeave={(e) => (e.target as HTMLElement).style.borderColor = '#d1d5db'}
                >
                  <svg style={{ margin: '0 auto', height: '3rem', width: '3rem', color: '#9ca3af', marginBottom: '1rem' }} stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <p style={{ fontSize: '0.875rem', color: '#4b5563', marginBottom: '0.25rem', margin: 0 }}>Upload your photo</p>
                  <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>PNG, JPG up to 10MB</p>
                </div>
              )}
              <input
                ref={personImageRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'person')}
                style={{ display: 'none' }}
              />
            </div>

            {/* Clothing Upload */}
            <div style={{ 
              backgroundColor: '#ffffff', 
              borderRadius: '0.75rem', 
              border: '1px solid #e5e7eb', 
              padding: '1.5rem' 
            }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '500', color: '#111827', marginBottom: '1rem' }}>Clothing Item</h3>
              {clothingImage ? (
                <div style={{ position: 'relative' }}>
                  <img
                    src={URL.createObjectURL(clothingImage)}
                    alt="Clothing"
                    style={{ 
                      width: '100%', 
                      height: '16rem', 
                      objectFit: 'cover', 
                      borderRadius: '0.5rem' 
                    }}
                  />
                  <button
                    onClick={() => clothingImageRef.current?.click()}
                    style={{ 
                      position: 'absolute', 
                      top: '0.5rem', 
                      right: '0.5rem', 
                      backgroundColor: '#ffffff', 
                      borderRadius: '50%', 
                      padding: '0.5rem', 
                      boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)', 
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'all 0.2s'
                    }}
                  >
                    <svg style={{ width: '1rem', height: '1rem', color: '#4b5563' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </button>
                  <div style={{ 
                    position: 'absolute', 
                    bottom: '0.5rem', 
                    left: '0.5rem', 
                    backgroundColor: 'rgba(0,0,0,0.5)', 
                    color: '#ffffff', 
                    fontSize: '0.75rem', 
                    padding: '0.25rem 0.5rem', 
                    borderRadius: '0.25rem' 
                  }}>
                    {clothingImage.name}
                  </div>
                </div>
              ) : (
                <div 
                  onClick={() => clothingImageRef.current?.click()}
                  style={{ 
                    border: '2px dashed #d1d5db', 
                    borderRadius: '0.5rem', 
                    padding: '3rem', 
                    textAlign: 'center', 
                    cursor: 'pointer',
                    transition: 'border-color 0.2s'
                  }}
                  onMouseEnter={(e) => (e.target as HTMLElement).style.borderColor = '#9ca3af'}
                  onMouseLeave={(e) => (e.target as HTMLElement).style.borderColor = '#d1d5db'}
                >
                  <svg style={{ margin: '0 auto', height: '3rem', width: '3rem', color: '#9ca3af', marginBottom: '1rem' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zM21 5a2 2 0 00-2-2h-4a2 2 0 00-2 2v12a4 4 0 004 4h4a4 4 0 004-4V5z" />
                  </svg>
                  <p style={{ fontSize: '0.875rem', color: '#4b5563', marginBottom: '0.25rem', margin: 0 }}>Upload clothing item</p>
                  <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>PNG, JPG up to 10MB</p>
                </div>
              )}
              <input
                ref={clothingImageRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'clothing')}
                style={{ display: 'none' }}
              />
            </div>
          </div>

          {/* Body Measurements */}
          <div style={{ 
            backgroundColor: '#ffffff', 
            borderRadius: '0.75rem', 
            border: '1px solid #e5e7eb', 
            padding: '1.5rem', 
            marginBottom: '2rem' 
          }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '500', color: '#111827', marginBottom: '1rem' }}>Body Measurements</h3>
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: window.innerWidth > 640 ? 'repeat(2, 1fr)' : '1fr', 
              gap: '1rem' 
            }}>
              <div>
                <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: '500', color: '#374151', marginBottom: '0.5rem' }}>Height (cm)</label>
                <input
                  type="number"
                  value={measurements.height}
                  onChange={(e) => setMeasurements(prev => ({ ...prev, height: Number(e.target.value) }))}
                  style={{ 
                    width: '100%', 
                    padding: '0.5rem 0.75rem', 
                    border: '1px solid #d1d5db', 
                    borderRadius: '0.5rem', 
                    fontSize: '0.875rem',
                    outline: 'none'
                  }}
                  min="140"
                  max="220"
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '0.875rem', fontWeight: '500', color: '#374151', marginBottom: '0.5rem' }}>Weight (kg)</label>
                <input
                  type="number"
                  value={measurements.weight}
                  onChange={(e) => setMeasurements(prev => ({ ...prev, weight: Number(e.target.value) }))}
                  style={{ 
                    width: '100%', 
                    padding: '0.5rem 0.75rem', 
                    border: '1px solid #d1d5db', 
                    borderRadius: '0.5rem', 
                    fontSize: '0.875rem',
                    outline: 'none'
                  }}
                  min="30"
                  max="150"
                />
              </div>
            </div>
          </div>

          {/* Generate Button */}
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
            <button
              onClick={processVirtualTryOn}
              disabled={!personImage || !clothingImage || isProcessing}
              style={{ 
                backgroundColor: (!personImage || !clothingImage || isProcessing) ? '#d1d5db' : '#000000',
                color: '#ffffff', 
                padding: '0.75rem 2rem', 
                borderRadius: '0.5rem', 
                fontWeight: '500', 
                border: 'none',
                cursor: (!personImage || !clothingImage || isProcessing) ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s',
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.5rem'
              }}
              onMouseEnter={(e) => {
                const target = e.target as HTMLButtonElement;
                if (!target.disabled) {
                  target.style.backgroundColor = '#1f2937';
                }
              }}
              onMouseLeave={(e) => {
                const target = e.target as HTMLButtonElement;
                if (!target.disabled) {
                  target.style.backgroundColor = '#000000';
                }
              }}
            >
              {isProcessing ? (
                <>
                  <div style={{ 
                    width: '1rem', 
                    height: '1rem', 
                    border: '2px solid #ffffff', 
                    borderTop: '2px solid transparent', 
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  <span>Processing...</span>
                </>
              ) : (
                <>
                  <svg style={{ width: '1rem', height: '1rem' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Generate Try-On</span>
                </>
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div style={{ 
              backgroundColor: '#fef2f2', 
              border: '1px solid #fecaca', 
              borderRadius: '0.5rem', 
              padding: '1rem', 
              marginBottom: '2rem' 
            }}>
              <div style={{ display: 'flex' }}>
                <svg style={{ flexShrink: 0, height: '1.25rem', width: '1.25rem', color: '#f87171' }} viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div style={{ marginLeft: '0.75rem' }}>
                  <h3 style={{ fontSize: '0.875rem', fontWeight: '500', color: '#991b1b', margin: 0 }}>Error</h3>
                  <p style={{ fontSize: '0.875rem', color: '#b91c1c', marginTop: '0.25rem', margin: 0 }}>{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Results */}
          {result && (
            <div style={{ 
              backgroundColor: '#ffffff', 
              borderRadius: '0.75rem', 
              border: '1px solid #e5e7eb', 
              padding: '1.5rem' 
            }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '500', color: '#111827', marginBottom: '1.5rem' }}>Your Virtual Try-On</h3>
              
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: window.innerWidth > 1024 ? 'repeat(2, 1fr)' : '1fr', 
                gap: '2rem' 
              }}>
                {/* Result Image */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <img
                    src={`data:image/jpeg;base64,${result.fitted_image}`}
                    alt="Virtual try-on result"
                    style={{ width: '100%', borderRadius: '0.5rem', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
                  />
                  <div style={{ display: 'flex', gap: '0.75rem' }}>
                    <button style={{ 
                      flex: 1, 
                      backgroundColor: '#f3f4f6', 
                      color: '#374151', 
                      padding: '0.5rem 1rem', 
                      borderRadius: '0.5rem', 
                      fontWeight: '500', 
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'background-color 0.2s'
                    }}>
                      Download
                    </button>
                    <button style={{ 
                      flex: 1, 
                      backgroundColor: '#000000', 
                      color: '#ffffff', 
                      padding: '0.5rem 1rem', 
                      borderRadius: '0.5rem', 
                      fontWeight: '500', 
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'background-color 0.2s'
                    }}>
                      Share
                    </button>
                  </div>
                </div>

                {/* Analysis */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  {/* Fit Scores */}
                  <div>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827', marginBottom: '1rem' }}>Fit Analysis</h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                          <span style={{ color: '#4b5563' }}>Fit Score</span>
                          <span style={{ fontWeight: '500' }}>{Math.round(result.fit_score * 100)}%</span>
                        </div>
                        <div style={{ width: '100%', backgroundColor: '#e5e7eb', borderRadius: '9999px', height: '0.5rem' }}>
                          <div 
                            style={{ 
                              backgroundColor: '#22c55e', 
                              height: '0.5rem', 
                              borderRadius: '9999px', 
                              transition: 'width 0.5s',
                              width: `${result.fit_score * 100}%`
                            }}
                          ></div>
                        </div>
                      </div>
                      <div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                          <span style={{ color: '#4b5563' }}>Confidence</span>
                          <span style={{ fontWeight: '500' }}>{Math.round(result.confidence * 100)}%</span>
                        </div>
                        <div style={{ width: '100%', backgroundColor: '#e5e7eb', borderRadius: '9999px', height: '0.5rem' }}>
                          <div 
                            style={{ 
                              backgroundColor: '#3b82f6', 
                              height: '0.5rem', 
                              borderRadius: '9999px', 
                              transition: 'width 0.5s',
                              width: `${result.confidence * 100}%`
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Details */}
                  <div>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827', marginBottom: '1rem' }}>Details</h4>
                    <div style={{ backgroundColor: '#f9fafb', borderRadius: '0.5rem', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                        <span style={{ color: '#4b5563' }}>Category</span>
                        <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>{result.clothing_analysis.category}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                        <span style={{ color: '#4b5563' }}>Style</span>
                        <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>{result.clothing_analysis.style}</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                        <span style={{ color: '#4b5563' }}>Processing Time</span>
                        <span style={{ fontWeight: '500' }}>{result.processing_time}s</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                        <span style={{ color: '#4b5563' }}>BMI</span>
                        <span style={{ fontWeight: '500' }}>{result.measurements.bmi}</span>
                      </div>
                    </div>
                  </div>

                  {/* Recommendations */}
                  {result.recommendations && result.recommendations.length > 0 && (
                    <div>
                      <h4 style={{ fontSize: '0.875rem', fontWeight: '500', color: '#111827', marginBottom: '1rem' }}>AI Recommendations</h4>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {result.recommendations.map((rec, index) => (
                          <div key={index} style={{ backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '0.5rem', padding: '0.75rem' }}>
                            <p style={{ fontSize: '0.875rem', color: '#1e40af', margin: 0 }}>{rec}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Instructions */}
          {!result && !isProcessing && (
            <div style={{ 
              backgroundColor: '#ffffff', 
              borderRadius: '0.75rem', 
              border: '1px solid #e5e7eb', 
              padding: '1.5rem' 
            }}>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '500', color: '#111827', marginBottom: '1rem' }}>How it works</h3>
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: window.innerWidth > 768 ? 'repeat(3, 1fr)' : '1fr', 
                gap: '1.5rem' 
              }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ 
                    width: '3rem', 
                    height: '3rem', 
                    backgroundColor: '#f3f4f6', 
                    borderRadius: '0.5rem', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    margin: '0 auto 0.75rem' 
                  }}>
                    <span style={{ fontSize: '1.125rem', fontWeight: '600', color: '#4b5563' }}>1</span>
                  </div>
                  <h4 style={{ fontWeight: '500', color: '#111827', marginBottom: '0.5rem' }}>Upload Photos</h4>
                  <p style={{ fontSize: '0.875rem', color: '#4b5563' }}>Upload a clear photo of yourself and the clothing item you want to try on.</p>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ 
                    width: '3rem', 
                    height: '3rem', 
                    backgroundColor: '#f3f4f6', 
                    borderRadius: '0.5rem', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    margin: '0 auto 0.75rem' 
                  }}>
                    <span style={{ fontSize: '1.125rem', fontWeight: '600', color: '#4b5563' }}>2</span>
                  </div>
                  <h4 style={{ fontWeight: '500', color: '#111827', marginBottom: '0.5rem' }}>Add Measurements</h4>
                  <p style={{ fontSize: '0.875rem', color: '#4b5563' }}>Enter your height and weight for accurate size matching.</p>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ 
                    width: '3rem', 
                    height: '3rem', 
                    backgroundColor: '#f3f4f6', 
                    borderRadius: '0.5rem', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    margin: '0 auto 0.75rem' 
                  }}>
                    <span style={{ fontSize: '1.125rem', fontWeight: '600', color: '#4b5563' }}>3</span>
                  </div>
                  <h4 style={{ fontWeight: '500', color: '#111827', marginBottom: '0.5rem' }}>Get Results</h4>
                  <p style={{ fontSize: '0.875rem', color: '#4b5563' }}>See how the clothing looks on you with AI-powered fitting analysis.</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* CSS Animation */}
      <style>{`
        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
};

export default App;