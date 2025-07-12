import React, { useState, useRef, useCallback } from 'react';
import { Upload, Camera, Zap, Activity, CheckCircle, AlertCircle, Download, Share2 } from 'lucide-react';

interface ProcessingStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  description: string;
  progress?: number;
}

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

const MyClosetAI: React.FC = () => {
  // 상태 관리
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [clothingImage, setClothingImage] = useState<File | null>(null);
  const [measurements, setMeasurements] = useState<UserMeasurements>({
    height: 170,
    weight: 65
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<TryOnResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);

  // Refs
  const personImageRef = useRef<HTMLInputElement>(null);
  const clothingImageRef = useRef<HTMLInputElement>(null);

  // 처리 단계 초기화
  const initializeSteps = (): ProcessingStep[] => {
    return [
      {
        id: 'upload',
        name: '이미지 업로드',
        status: 'pending',
        description: '사용자와 의류 이미지를 서버로 전송합니다'
      },
      {
        id: 'segmentation',
        name: '신체 분석',
        status: 'pending',
        description: '얼굴, 신체 부위를 OpenCV로 정확히 인식합니다'
      },
      {
        id: 'measurement',
        name: '치수 측정',
        status: 'pending',
        description: '신체 치수를 AI로 정밀 추정합니다'
      },
      {
        id: 'clothing',
        name: '의류 분석',
        status: 'pending',
        description: '의류 카테고리, 색상, 스타일을 분석합니다'
      },
      {
        id: 'fitting',
        name: '가상 피팅',
        status: 'pending',
        description: 'AI가 의류를 자연스럽게 착용시킵니다'
      }
    ];
  };

  // 이미지 업로드 핸들러
  const handleImageUpload = useCallback((file: File, type: 'person' | 'clothing') => {
    if (type === 'person') {
      setPersonImage(file);
    } else {
      setClothingImage(file);
    }
    setError(null);
  }, []);

  // 드래그 앤 드롭 핸들러
  const handleDrop = useCallback((e: React.DragEvent, type: 'person' | 'clothing') => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    const imageFile = files.find(file => file.type.startsWith('image/'));
    
    if (imageFile) {
      handleImageUpload(imageFile, type);
    }
  }, [handleImageUpload]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  // 처리 단계 업데이트
  const updateStepStatus = (stepId: string, status: ProcessingStep['status'], progress?: number) => {
    setProcessingSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, status, progress } : step
    ));
  };

  // 메인 처리 함수
  const processVirtualTryOn = async () => {
    if (!personImage || !clothingImage) {
      setError('신체 사진과 의류 사진을 모두 업로드해주세요.');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);
    
    const steps = initializeSteps();
    setProcessingSteps(steps);

    try {
      // 1단계: 업로드
      updateStepStatus('upload', 'processing');
      const formData = new FormData();
      formData.append('person_image', personImage);
      formData.append('clothing_image', clothingImage);
      formData.append('height', measurements.height.toString());
      formData.append('weight', measurements.weight.toString());
      updateStepStatus('upload', 'completed');

      // 나머지 단계들을 순차적으로 처리
      const stepIds = ['segmentation', 'measurement', 'clothing', 'fitting'];
      
      for (const stepId of stepIds) {
        updateStepStatus(stepId, 'processing');
        await new Promise(resolve => setTimeout(resolve, 800)); // 시각적 효과
      }

      // 실제 API 호출
      const response = await fetch('http://localhost:8000/api/virtual-tryon', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`서버 오류: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.success) {
        // 모든 단계 완료
        stepIds.forEach(stepId => updateStepStatus(stepId, 'completed'));
        setResult(data);
      } else {
        throw new Error(data.error || '처리 중 오류가 발생했습니다.');
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : '처리 중 오류가 발생했습니다.');
      // 현재 처리 중인 단계를 오류로 표시
      setProcessingSteps(prev => prev.map(step => 
        step.status === 'processing' ? { ...step, status: 'error' } : step
      ));
    } finally {
      setIsProcessing(false);
    }
  };

  // 파일 입력 클릭
  const triggerFileInput = (type: 'person' | 'clothing') => {
    if (type === 'person') {
      personImageRef.current?.click();
    } else {
      clothingImageRef.current?.click();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-pink-50">
      {/* 헤더 */}
      <header className="bg-white/80 backdrop-blur-md shadow-sm border-b border-purple-100">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl flex items-center justify-center">
                <Camera className="text-white w-6 h-6" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                  MyCloset AI
                </h1>
                <p className="text-gray-600 text-sm">AI 가상 피팅 시스템</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Activity className="w-4 h-4 text-green-500" />
              <span>서버 연결됨</span>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8">
          {/* 좌측: 입력 영역 */}
          <div className="space-y-6">
            {/* 이미지 업로드 섹션 */}
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-purple-100">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">📸 이미지 업로드</h2>
              
              <div className="grid grid-cols-2 gap-4">
                {/* 사용자 이미지 */}
                <div 
                  className="border-2 border-dashed border-purple-300 rounded-xl p-4 text-center cursor-pointer hover:border-purple-400 transition-colors bg-purple-25"
                  onDrop={(e) => handleDrop(e, 'person')}
                  onDragOver={handleDragOver}
                  onClick={() => triggerFileInput('person')}
                >
                  {personImage ? (
                    <div>
                      <img 
                        src={URL.createObjectURL(personImage)} 
                        alt="사용자 이미지" 
                        className="w-full h-40 object-cover rounded-lg mb-2"
                      />
                      <p className="text-sm text-gray-600">{personImage.name}</p>
                    </div>
                  ) : (
                    <div className="py-8">
                      <Upload className="w-8 h-8 text-purple-400 mx-auto mb-2" />
                      <p className="text-gray-600 text-sm">사용자 사진</p>
                      <p className="text-gray-400 text-xs">클릭하거나 드래그하세요</p>
                    </div>
                  )}
                </div>

                {/* 의류 이미지 */}
                <div 
                  className="border-2 border-dashed border-blue-300 rounded-xl p-4 text-center cursor-pointer hover:border-blue-400 transition-colors bg-blue-25"
                  onDrop={(e) => handleDrop(e, 'clothing')}
                  onDragOver={handleDragOver}
                  onClick={() => triggerFileInput('clothing')}
                >
                  {clothingImage ? (
                    <div>
                      <img 
                        src={URL.createObjectURL(clothingImage)} 
                        alt="의류 이미지" 
                        className="w-full h-40 object-cover rounded-lg mb-2"
                      />
                      <p className="text-sm text-gray-600">{clothingImage.name}</p>
                    </div>
                  ) : (
                    <div className="py-8">
                      <Upload className="w-8 h-8 text-blue-400 mx-auto mb-2" />
                      <p className="text-gray-600 text-sm">의류 사진</p>
                      <p className="text-gray-400 text-xs">클릭하거나 드래그하세요</p>
                    </div>
                  )}
                </div>
              </div>

              {/* 숨겨진 파일 입력 */}
              <input
                ref={personImageRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'person')}
                className="hidden"
              />
              <input
                ref={clothingImageRef}
                type="file"
                accept="image/*"
                onChange={(e) => e.target.files?.[0] && handleImageUpload(e.target.files[0], 'clothing')}
                className="hidden"
              />
            </div>

            {/* 측정 정보 섹션 */}
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-purple-100">
              <h2 className="text-xl font-semibold mb-4 text-gray-800">📏 신체 정보</h2>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">키 (cm)</label>
                  <input
                    type="number"
                    value={measurements.height}
                    onChange={(e) => setMeasurements(prev => ({ ...prev, height: Number(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    min="140"
                    max="220"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">몸무게 (kg)</label>
                  <input
                    type="number"
                    value={measurements.weight}
                    onChange={(e) => setMeasurements(prev => ({ ...prev, weight: Number(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    min="30"
                    max="150"
                  />
                </div>
              </div>
            </div>

            {/* 실행 버튼 */}
            <button
              onClick={processVirtualTryOn}
              disabled={!personImage || !clothingImage || isProcessing}
              className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-4 px-6 rounded-xl font-semibold text-lg shadow-lg hover:shadow-xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {isProcessing ? (
                <>
                  <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full"></div>
                  <span>AI 처리 중...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  <span>가상 피팅 시작</span>
                </>
              )}
            </button>

            {/* 오류 메시지 */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-center space-x-3">
                <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0" />
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* 우측: 결과 영역 */}
          <div className="space-y-6">
            {/* 처리 단계 */}
            {isProcessing && (
              <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-purple-100">
                <h3 className="text-lg font-semibold mb-4 text-gray-800">🔄 처리 단계</h3>
                
                <div className="space-y-3">
                  {processingSteps.map((step) => (
                    <div key={step.id} className="flex items-center space-x-3">
                      <div className="flex-shrink-0">
                        {step.status === 'completed' && (
                          <CheckCircle className="w-5 h-5 text-green-500" />
                        )}
                        {step.status === 'processing' && (
                          <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                        )}
                        {step.status === 'error' && (
                          <AlertCircle className="w-5 h-5 text-red-500" />
                        )}
                        {step.status === 'pending' && (
                          <div className="w-5 h-5 border-2 border-gray-300 rounded-full"></div>
                        )}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className={`font-medium ${
                            step.status === 'completed' ? 'text-green-700' :
                            step.status === 'processing' ? 'text-blue-700' :
                            step.status === 'error' ? 'text-red-700' :
                            'text-gray-500'
                          }`}>
                            {step.name}
                          </span>
                        </div>
                        <p className="text-xs text-gray-600">{step.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 결과 표시 */}
            {result && (
              <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-purple-100">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold text-gray-800">✨ 가상 피팅 결과</h3>
                  <div className="flex space-x-2">
                    <button className="p-2 bg-blue-100 hover:bg-blue-200 rounded-lg transition-colors">
                      <Download className="w-4 h-4 text-blue-600" />
                    </button>
                    <button className="p-2 bg-green-100 hover:bg-green-200 rounded-lg transition-colors">
                      <Share2 className="w-4 h-4 text-green-600" />
                    </button>
                  </div>
                </div>

                {/* 결과 이미지 */}
                <div className="mb-4">
                  <img
                    src={`data:image/jpeg;base64,${result.fitted_image}`}
                    alt="가상 피팅 결과"
                    className="w-full rounded-xl shadow-md"
                  />
                </div>

                {/* 분석 결과 */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-3">
                    <h4 className="font-semibold text-sm text-gray-700 mb-1">핏 점수</h4>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${result.fit_score * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-bold text-gray-700">
                        {Math.round(result.fit_score * 100)}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-3">
                    <h4 className="font-semibold text-sm text-gray-700 mb-1">신뢰도</h4>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${result.confidence * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-bold text-gray-700">
                        {Math.round(result.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* 추천사항 */}
                <div>
                  <h4 className="font-semibold text-sm text-gray-700 mb-2">💡 AI 추천</h4>
                  <div className="space-y-2">
                    {result.recommendations.map((rec, index) => (
                      <div key={index} className="bg-amber-50 border border-amber-200 rounded-lg p-2">
                        <p className="text-sm text-amber-800">{rec}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* 기술 정보 */}
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>처리 시간: {result.processing_time}초</span>
                    <span>의류: {result.clothing_analysis.category}</span>
                    <span>BMI: {result.measurements.bmi}</span>
                  </div>
                </div>
              </div>
            )}

            {/* 기본 안내 */}
            {!isProcessing && !result && (
              <div className="bg-gradient-to-br from-purple-100 to-blue-100 rounded-2xl p-8 text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Camera className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-2 text-gray-800">AI 가상 피팅 준비</h3>
                <p className="text-gray-600 mb-4">
                  사진을 업로드하고 신체 정보를 입력하면<br />
                  AI가 완벽한 가상 피팅을 제공합니다
                </p>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div className="text-center">
                    <div className="w-8 h-8 bg-purple-200 rounded-full flex items-center justify-center mx-auto mb-1">
                      <span className="text-purple-600 font-bold">1</span>
                    </div>
                    <p className="text-gray-600">사진 업로드</p>
                  </div>
                  <div className="text-center">
                    <div className="w-8 h-8 bg-blue-200 rounded-full flex items-center justify-center mx-auto mb-1">
                      <span className="text-blue-600 font-bold">2</span>
                    </div>
                    <p className="text-gray-600">AI 분석</p>
                  </div>
                  <div className="text-center">
                    <div className="w-8 h-8 bg-green-200 rounded-full flex items-center justify-center mx-auto mb-1">
                      <span className="text-green-600 font-bold">3</span>
                    </div>
                    <p className="text-gray-600">결과 확인</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MyClosetAI;