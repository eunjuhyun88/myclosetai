import React, { useState, useRef, useEffect } from 'react';
import { usePipeline, usePipelineHealth } from './hooks/usePipeline';
import { fileUtils, devTools } from './services/api';

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

// 8단계 정의
const PIPELINE_STEPS = [
  { id: 1, name: '이미지 업로드', description: '사용자 사진과 의류 이미지를 업로드합니다' },
  { id: 2, name: '신체 측정', description: '키와 몸무게 등 신체 정보를 입력합니다' },
  { id: 3, name: '인체 파싱', description: 'AI가 신체 부위를 20개 영역으로 분석합니다' },
  { id: 4, name: '포즈 추정', description: '18개 키포인트로 자세를 분석합니다' },
  { id: 5, name: '의류 분석', description: '의류 스타일과 색상을 분석합니다' },
  { id: 6, name: '기하학적 매칭', description: '신체와 의류를 정확히 매칭합니다' },
  { id: 7, name: '가상 피팅', description: 'AI로 가상 착용 결과를 생성합니다' },
  { id: 8, name: '결과 확인', description: '최종 결과를 확인하고 저장합니다' }
];

const App: React.FC = () => {
  // 현재 단계 관리
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  
  // 파일 상태
  const [personImage, setPersonImage] = useState<File | null>(null);
  const [clothingImage, setClothingImage] = useState<File | null>(null);
  const [measurements, setMeasurements] = useState<UserMeasurements>({
    height: 170,
    weight: 65
  });

  // 단계별 결과 저장
  const [stepResults, setStepResults] = useState<{[key: number]: any}>({});
  
  // 파일 검증 에러
  const [fileErrors, setFileErrors] = useState<{
    person?: string;
    clothing?: string;
  }>({});

  // 파일 참조
  const personImageRef = useRef<HTMLInputElement>(null);
  const clothingImageRef = useRef<HTMLInputElement>(null);

  // 파이프라인 훅 사용 (수정된 설정)
  const {
    isProcessing,
    progress,
    progressMessage,
    currentStep: pipelineStep,
    result,
    error,
    processVirtualTryOn,
    clearError,
    clearResult,
    reset,
    testConnection,
    warmupPipeline,
    cancelCurrentRequest
  } = usePipeline({
    baseURL: 'http://localhost:8000',
    autoHealthCheck: true,
    healthCheckInterval: 30000,
    requestTimeout: 60000, // 60초로 증가
    enableDebugMode: true,
    enableRetry: true,
    maxRetryAttempts: 3
  });

  // 헬스체크 훅
  const { isHealthy, isChecking } = usePipelineHealth({
    baseURL: 'http://localhost:8000',
    autoHealthCheck: true,
    healthCheckInterval: 30000
  });

  // 파일 업로드 핸들러 (수정된 버전)
  const handleImageUpload = (file: File, type: 'person' | 'clothing') => {
    // 파일 검증
    const validation = fileUtils.validateImageFile(file);
    
    if (!validation.valid) {
      setFileErrors(prev => ({
        ...prev,
        [type]: validation.error
      }));
      return;
    }

    // 에러 초기화
    setFileErrors(prev => ({
      ...prev,
      [type]: undefined
    }));

    if (type === 'person') {
      setPersonImage(file);
      console.log('✅ 사용자 이미지 업로드:', {
        name: file.name,
        size: fileUtils.formatFileSize(file.size),
        type: file.type
      });
    } else {
      setClothingImage(file);
      console.log('✅ 의류 이미지 업로드:', {
        name: file.name,
        size: fileUtils.formatFileSize(file.size),
        type: file.type
      });
    }
    
    clearError();
  };

  // 다음 단계로 이동
  const goToNextStep = () => {
    if (currentStep < 8) {
      setCompletedSteps(prev => [...prev, currentStep]);
      setCurrentStep(prev => prev + 1);
    }
  };

  // 이전 단계로 이동
  const goToPreviousStep = () => {
    if (currentStep > 1) {
      setCurrentStep(prev => prev - 1);
      setCompletedSteps(prev => prev.filter(step => step < currentStep - 1));
    }
  };

  // 단계별 처리 함수 (수정된 버전)
  const processCurrentStep = async () => {
    try {
      switch (currentStep) {
        case 1: // 이미지 업로드 완료 확인
          if (!personImage || !clothingImage) {
            alert('사용자 이미지와 의류 이미지를 모두 업로드해주세요.');
            return;
          }
          
          // 파일 재검증
          const personValidation = fileUtils.validateImageFile(personImage);
          const clothingValidation = fileUtils.validateImageFile(clothingImage);
          
          if (!personValidation.valid || !clothingValidation.valid) {
            alert('업로드된 파일에 문제가 있습니다. 다시 확인해주세요.');
            return;
          }
          
          goToNextStep();
          break;

        case 2: // 신체 측정 완료 확인
          if (measurements.height <= 0 || measurements.weight <= 0) {
            alert('올바른 키와 몸무게를 입력해주세요.');
            return;
          }
          
          if (measurements.height < 100 || measurements.height > 250) {
            alert('키는 100cm~250cm 범위로 입력해주세요.');
            return;
          }
          
          if (measurements.weight < 30 || measurements.weight > 300) {
            alert('몸무게는 30kg~300kg 범위로 입력해주세요.');
            return;
          }
          
          goToNextStep();
          break;

        case 3: // 인체 파싱 시작
          await processStepWithAPI('human_parsing', '인체 파싱을 수행합니다...');
          break;

        case 4: // 포즈 추정 시작
          await processStepWithAPI('pose_estimation', '포즈를 분석합니다...');
          break;

        case 5: // 의류 분석 시작
          await processStepWithAPI('clothing_analysis', '의류를 분석합니다...');
          break;

        case 6: // 기하학적 매칭 시작
          await processStepWithAPI('geometric_matching', '신체와 의류를 매칭합니다...');
          break;

        case 7: // 가상 피팅 시작
          await processVirtualFitting();
          break;

        case 8: // 결과 확인 - 완료
          alert('가상 피팅이 완료되었습니다!');
          break;
      }
    } catch (error: any) {
      console.error('단계 처리 중 오류:', error);
      alert(`단계 처리 중 오류가 발생했습니다: ${error.message}`);
    }
  };

  // API를 통한 단계별 처리
  const processStepWithAPI = async (stepType: string, message: string) => {
    if (!personImage || !clothingImage) return;

    // 시뮬레이션: 실제로는 각 단계별 API를 호출
    const stepResult = await simulateStepProcessing(stepType, message);
    setStepResults(prev => ({ ...prev, [currentStep]: stepResult }));
    
    // 자동으로 다음 단계로 이동
    setTimeout(() => {
      goToNextStep();
    }, 1500);
  };

  // 가상 피팅 전체 처리 (수정된 버전)
  const processVirtualFitting = async () => {
    if (!personImage || !clothingImage) {
      alert('이미지 파일이 누락되었습니다.');
      return;
    }

    try {
      console.log('🎯 가상 피팅 API 호출 시작');
      
      const response = await processVirtualTryOn({
        person_image: personImage,
        clothing_image: clothingImage,
        height: measurements.height,
        weight: measurements.weight,
        quality_mode: 'balanced',
        enable_realtime: true,
        session_id: `app_session_${Date.now()}`
      });

      console.log('✅ 가상 피팅 API 응답:', response);

      // 처리 완료 후 다음 단계로
      setTimeout(() => {
        goToNextStep();
      }, 2000);

    } catch (error: any) {
      console.error('❌ 가상 피팅 실패:', error);
      alert(`가상 피팅 처리 중 오류가 발생했습니다: ${error.message}`);
    }
  };

  // 단계 처리 시뮬레이션
  const simulateStepProcessing = async (stepType: string, message: string): Promise<any> => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          stepType,
          success: true,
          confidence: 0.85 + Math.random() * 0.1,
          processing_time: 0.5 + Math.random() * 1.0,
          message: `${message} 완료`
        });
      }, 1000 + Math.random() * 2000);
    });
  };

  // 현재 단계가 완료 가능한지 확인
  const canProceedToNext = () => {
    switch (currentStep) {
      case 1:
        return personImage && clothingImage && 
               !fileErrors.person && !fileErrors.clothing;
      case 2:
        return measurements.height > 0 && measurements.weight > 0 &&
               measurements.height >= 100 && measurements.height <= 250 &&
               measurements.weight >= 30 && measurements.weight <= 300;
      case 3:
      case 4:
      case 5:
      case 6:
        return stepResults[currentStep]?.success;
      case 7:
        return result?.success;
      case 8:
        return true;
      default:
        return false;
    }
  };

  // 서버 상태 색상
  const getServerStatusColor = () => {
    if (isChecking) return '#f59e0b';
    return isHealthy ? '#4ade80' : '#ef4444';
  };

  const getServerStatusText = () => {
    if (isChecking) return 'Checking...';
    return isHealthy ? 'Server Online' : 'Server Offline';
  };

  // 웹소켓 연결 테스트 (수정된 버전)
  const handleTestConnection = async () => {
    try {
      await testConnection();
    } catch (error) {
      console.error('연결 테스트 실패:', error);
    }
  };

  // 파이프라인 워밍업 (수정된 버전)
  const handleWarmup = async () => {
    try {
      await warmupPipeline('balanced');
    } catch (error) {
      console.error('워밍업 실패:', error);
    }
  };

  // 개발 도구 함수들
  const handleDevTest = async () => {
    const result = await devTools.testAPI();
    console.log('개발 도구 테스트 결과:', result);
    alert(result.message);
  };

  const handleDummyTest = async () => {
    try {
      const result = await devTools.testDummyVirtualTryOn();
      console.log('더미 테스트 결과:', result);
      alert(result.message);
    } catch (error: any) {
      console.error('더미 테스트 실패:', error);
      alert(`더미 테스트 실패: ${error.message}`);
    }
  };

  // 요청 취소 핸들러
  const handleCancelRequest = () => {
    if (isProcessing) {
      cancelCurrentRequest();
      reset();
      alert('요청이 취소되었습니다.');
    }
  };

  // 파일 드래그 앤 드롭 지원
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent, type: 'person' | 'clothing') => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      handleImageUpload(files[0], type);
    }
  };

  // 단계별 컨텐츠 렌더링
  const renderStepContent = () => {
    switch (currentStep) {
      case 1:
        return renderImageUploadStep();
      case 2:
        return renderMeasurementsStep();
      case 3:
      case 4:
      case 5:
      case 6:
        return renderProcessingStep();
      case 7:
        return renderVirtualFittingStep();
      case 8:
        return renderResultStep();
      default:
        return null;
    }
  };

  const renderImageUploadStep = () => (
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
        border: fileErrors.person ? '2px solid #ef4444' : '1px solid #e5e7eb', 
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
              {personImage.name} ({fileUtils.formatFileSize(personImage.size)})
            </div>
          </div>
        ) : (
          <div 
            onClick={() => personImageRef.current?.click()}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, 'person')}
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
            <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>PNG, JPG, WebP up to 50MB</p>
            <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>또는 드래그 앤 드롭</p>
          </div>
        )}
        {fileErrors.person && (
          <div style={{ 
            marginTop: '0.5rem', 
            padding: '0.5rem', 
            backgroundColor: '#fef2f2', 
            border: '1px solid #fecaca', 
            borderRadius: '0.25rem', 
            fontSize: '0.875rem', 
            color: '#b91c1c' 
          }}>
            {fileErrors.person}
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
        border: fileErrors.clothing ? '2px solid #ef4444' : '1px solid #e5e7eb', 
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
              {clothingImage.name} ({fileUtils.formatFileSize(clothingImage.size)})
            </div>
          </div>
        ) : (
          <div 
            onClick={() => clothingImageRef.current?.click()}
            onDragOver={handleDragOver}
            onDrop={(e) => handleDrop(e, 'clothing')}
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
            <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>PNG, JPG, WebP up to 50MB</p>
            <p style={{ fontSize: '0.75rem', color: '#6b7280', margin: 0 }}>또는 드래그 앤 드롭</p>
          </div>
        )}
        {fileErrors.clothing && (
          <div style={{ 
            marginTop: '0.5rem', 
            padding: '0.5rem', 
            backgroundColor: '#fef2f2', 
            border: '1px solid #fecaca', 
            borderRadius: '0.25rem', 
            fontSize: '0.875rem', 
            color: '#b91c1c' 
          }}>
            {fileErrors.clothing}
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
  );

  const renderMeasurementsStep = () => (
    <div style={{ 
      backgroundColor: '#ffffff', 
      borderRadius: '0.75rem', 
      border: '1px solid #e5e7eb', 
      padding: '1.5rem', 
      maxWidth: '28rem',
      margin: '0 auto'
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
            min="100"
            max="250"
            placeholder="170"
          />
          <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
            100-250cm 범위
          </div>
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
            max="300"
            placeholder="65"
          />
          <div style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
            30-300kg 범위
          </div>
        </div>
      </div>
      
      {/* BMI 계산 표시 */}
      {measurements.height > 0 && measurements.weight > 0 && (
        <div style={{ 
          marginTop: '1rem', 
          padding: '0.75rem', 
          backgroundColor: '#f9fafb', 
          borderRadius: '0.5rem' 
        }}>
          <div style={{ fontSize: '0.875rem', color: '#4b5563' }}>
            BMI: {(measurements.weight / Math.pow(measurements.height / 100, 2)).toFixed(1)}
          </div>
        </div>
      )}
    </div>
  );

  const renderProcessingStep = () => {
    const stepData = PIPELINE_STEPS[currentStep - 1];
    const stepResult = stepResults[currentStep];

    return (
      <div style={{ textAlign: 'center', maxWidth: '28rem', margin: '0 auto' }}>
        <div style={{ backgroundColor: '#ffffff', borderRadius: '0.75rem', border: '1px solid #e5e7eb', padding: '2rem' }}>
          <div style={{ marginBottom: '1.5rem' }}>
            <div style={{ 
              width: '4rem', 
              height: '4rem', 
              margin: '0 auto', 
              backgroundColor: '#eff6ff', 
              borderRadius: '50%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              marginBottom: '1rem' 
            }}>
              {stepResult?.success ? (
                <svg style={{ width: '2rem', height: '2rem', color: '#22c55e' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <div style={{ 
                  width: '2rem', 
                  height: '2rem', 
                  border: '4px solid #3b82f6', 
                  borderTop: '4px solid transparent', 
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}></div>
              )}
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827' }}>{stepData.name}</h3>
            <p style={{ color: '#4b5563', marginTop: '0.5rem' }}>{stepData.description}</p>
          </div>

          {stepResult && (
            <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f9fafb', borderRadius: '0.5rem' }}>
              <p style={{ fontSize: '0.875rem', color: '#4b5563' }}>{stepResult.message}</p>
              <p style={{ fontSize: '0.75rem', color: '#6b7280', marginTop: '0.25rem' }}>
                신뢰도: {(stepResult.confidence * 100).toFixed(1)}% | 
                처리시간: {stepResult.processing_time.toFixed(1)}초
              </p>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderVirtualFittingStep = () => (
    <div style={{ textAlign: 'center', maxWidth: '28rem', margin: '0 auto' }}>
      <div style={{ backgroundColor: '#ffffff', borderRadius: '0.75rem', border: '1px solid #e5e7eb', padding: '2rem' }}>
        <div style={{ marginBottom: '1.5rem' }}>
          <div style={{ 
            width: '4rem', 
            height: '4rem', 
            margin: '0 auto', 
            backgroundColor: '#f3e8ff', 
            borderRadius: '50%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            marginBottom: '1rem' 
          }}>
            {result?.success ? (
              <svg style={{ width: '2rem', height: '2rem', color: '#22c55e' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
            ) : (
              <div style={{ 
                width: '2rem', 
                height: '2rem', 
                border: '4px solid #7c3aed', 
                borderTop: '4px solid transparent', 
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></div>
            )}
          </div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827' }}>AI 가상 피팅 생성</h3>
          <p style={{ color: '#4b5563', marginTop: '0.5rem' }}>딥러닝 모델이 최종 결과를 생성하고 있습니다</p>
        </div>

        {isProcessing && (
          <div style={{ marginTop: '1rem' }}>
            <div style={{ 
              width: '100%', 
              backgroundColor: '#f3f4f6', 
              borderRadius: '0.5rem', 
              height: '0.75rem',
              marginBottom: '0.5rem'
            }}>
              <div 
                style={{ 
                  backgroundColor: '#7c3aed', 
                  height: '0.75rem', 
                  borderRadius: '0.5rem', 
                  transition: 'width 0.3s',
                  width: `${progress}%`
                }}
              ></div>
            </div>
            <p style={{ fontSize: '0.875rem', color: '#4b5563' }}>{progressMessage}</p>
            
            {/* 취소 버튼 */}
            <button
              onClick={handleCancelRequest}
              style={{
                marginTop: '1rem',
                padding: '0.5rem 1rem',
                backgroundColor: '#ef4444',
                color: '#ffffff',
                borderRadius: '0.5rem',
                border: 'none',
                cursor: 'pointer',
                fontSize: '0.875rem'
              }}
            >
              취소
            </button>
          </div>
        )}

        {result && (
          <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f0fdf4', borderRadius: '0.5rem' }}>
            <p style={{ fontSize: '0.875rem', color: '#15803d' }}>가상 피팅 완성!</p>
            <p style={{ fontSize: '0.75rem', color: '#16a34a', marginTop: '0.25rem' }}>
              품질 점수: {(result.fit_score * 100).toFixed(1)}% | 
              처리시간: {result.processing_time.toFixed(1)}초
            </p>
          </div>
        )}
      </div>
    </div>
  );

  const renderResultStep = () => {
    if (!result) return null;

    return (
      <div style={{ maxWidth: '64rem', margin: '0 auto' }}>
        <div style={{ backgroundColor: '#ffffff', borderRadius: '0.75rem', border: '1px solid #e5e7eb', padding: '1.5rem' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: '600', color: '#111827', marginBottom: '1.5rem', textAlign: 'center' }}>가상 피팅 결과</h3>
          
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: window.innerWidth > 1024 ? 'repeat(2, 1fr)' : '1fr', 
            gap: '2rem' 
          }}>
            {/* Result Image */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {result.fitted_image || result.result_image ? (
                <img
                  src={`data:image/jpeg;base64,${result.fitted_image || result.result_image}`}
                  alt="Virtual try-on result"
                  style={{ width: '100%', borderRadius: '0.5rem', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1)' }}
                />
              ) : (
                <div style={{ 
                  width: '100%', 
                  height: '20rem', 
                  backgroundColor: '#f3f4f6', 
                  borderRadius: '0.5rem', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  color: '#6b7280'
                }}>
                  결과 이미지 없음
                </div>
              )}
              
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button 
                  onClick={() => {
                    if (result.fitted_image || result.result_image) {
                      const link = document.createElement('a');
                      link.href = `data:image/jpeg;base64,${result.fitted_image || result.result_image}`;
                      link.download = 'virtual-tryon-result.jpg';
                      link.click();
                    }
                  }}
                  disabled={!result.fitted_image && !result.result_image}
                  style={{ 
                    flex: 1, 
                    backgroundColor: result.fitted_image || result.result_image ? '#f3f4f6' : '#e5e7eb', 
                    color: '#374151', 
                    padding: '0.5rem 1rem', 
                    borderRadius: '0.5rem', 
                    fontWeight: '500', 
                    border: 'none',
                    cursor: result.fitted_image || result.result_image ? 'pointer' : 'not-allowed',
                    transition: 'background-color 0.2s'
                  }}
                >
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
                    <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>
                      {result?.clothing_analysis?.category || 'Unknown'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                    <span style={{ color: '#4b5563' }}>Style</span>
                    <span style={{ fontWeight: '500', textTransform: 'capitalize' }}>
                      {result?.clothing_analysis?.style || 'Unknown'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                    <span style={{ color: '#4b5563' }}>Processing Time</span>
                    <span style={{ fontWeight: '500' }}>{result?.processing_time?.toFixed(1) || 0}s</span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                    <span style={{ color: '#4b5563' }}>BMI</span>
                    <span style={{ fontWeight: '500' }}>{result?.measurements?.bmi?.toFixed(1) || 0}</span>
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
      </div>
    );
  };

  // 컴포넌트 마운트 시 개발 도구 정보 출력
  useEffect(() => {
    console.log('🛠️ MyCloset AI App 시작됨');
    console.log('📋 개발 도구 사용법:');
    console.log('  - devTools.testAPI(): API 연결 테스트');
    console.log('  - devTools.testDummyVirtualTryOn(): 더미 가상 피팅 테스트');
    console.log('  - devTools.exportDebugInfo(): 디버그 정보 내보내기');
  }, []);

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
            
            {/* 서버 상태 및 개발 도구 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              {/* 개발 도구 버튼들 */}
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                <button
                  onClick={handleTestConnection}
                  disabled={isProcessing}
                  style={{
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    backgroundColor: '#e5e7eb',
                    color: '#374151',
                    border: 'none',
                    borderRadius: '0.25rem',
                    cursor: isProcessing ? 'not-allowed' : 'pointer',
                    opacity: isProcessing ? 0.5 : 1
                  }}
                >
                  Test
                </button>
                <button
                  onClick={handleWarmup}
                  disabled={isProcessing}
                  style={{
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    backgroundColor: '#e5e7eb',
                    color: '#374151',
                    border: 'none',
                    borderRadius: '0.25rem',
                    cursor: isProcessing ? 'not-allowed' : 'pointer',
                    opacity: isProcessing ? 0.5 : 1
                  }}
                >
                  Warmup
                </button>
                <button
                  onClick={handleDevTest}
                  disabled={isProcessing}
                  style={{
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    backgroundColor: '#e5e7eb',
                    color: '#374151',
                    border: 'none',
                    borderRadius: '0.25rem',
                    cursor: isProcessing ? 'not-allowed' : 'pointer',
                    opacity: isProcessing ? 0.5 : 1
                  }}
                >
                  DevTest
                </button>
                <button
                  onClick={handleDummyTest}
                  disabled={isProcessing}
                  style={{
                    padding: '0.25rem 0.5rem',
                    fontSize: '0.75rem',
                    backgroundColor: '#e5e7eb',
                    color: '#374151',
                    border: 'none',
                    borderRadius: '0.25rem',
                    cursor: isProcessing ? 'not-allowed' : 'pointer',
                    opacity: isProcessing ? 0.5 : 1
                  }}
                >
                  Dummy
                </button>
              </div>

              {/* 진행률 표시 (처리 중일 때) */}
              {isProcessing && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ 
                    width: '0.75rem', 
                    height: '0.75rem', 
                    border: '2px solid #3b82f6', 
                    borderTop: '2px solid transparent', 
                    borderRadius: '50%',
                    animation: 'spin 1s linear infinite'
                  }}></div>
                  <span style={{ fontSize: '0.875rem', color: '#4b5563' }}>
                    {progress}%
                  </span>
                </div>
              )}

              {/* 서버 상태 */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ 
                  height: '0.5rem', 
                  width: '0.5rem', 
                  backgroundColor: getServerStatusColor(),
                  borderRadius: '50%',
                  transition: 'background-color 0.3s'
                }}></div>
                <span style={{ fontSize: '0.875rem', color: '#4b5563' }}>
                  {getServerStatusText()}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ maxWidth: '80rem', margin: '0 auto', padding: '2rem 1rem' }}>
        {/* Progress Bar */}
        <div style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
            <h2 style={{ fontSize: '1.875rem', fontWeight: '700', color: '#111827', margin: 0 }}>AI Virtual Try-On</h2>
            <span style={{ fontSize: '0.875rem', color: '#4b5563' }}>Step {currentStep} of 8</span>
          </div>
          
          {/* Step Progress */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
            {PIPELINE_STEPS.map((step, index) => (
              <div key={step.id} style={{ display: 'flex', alignItems: 'center' }}>
                <div 
                  style={{
                    width: '2rem', 
                    height: '2rem', 
                    borderRadius: '50%', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    fontSize: '0.875rem', 
                    fontWeight: '500',
                    backgroundColor: completedSteps.includes(step.id) 
                      ? '#22c55e' 
                      : currentStep === step.id 
                        ? '#3b82f6' 
                        : '#e5e7eb',
                    color: completedSteps.includes(step.id) || currentStep === step.id 
                      ? '#ffffff' 
                      : '#4b5563'
                  }}
                >
                  {completedSteps.includes(step.id) ? (
                    <svg style={{ width: '1rem', height: '1rem' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    step.id
                  )}
                </div>
                {index < PIPELINE_STEPS.length - 1 && (
                  <div 
                    style={{
                      width: '3rem', 
                      height: '2px', 
                      marginLeft: '0.5rem', 
                      marginRight: '0.5rem',
                      backgroundColor: completedSteps.includes(step.id) ? '#22c55e' : '#e5e7eb'
                    }}
                  ></div>
                )}
              </div>
            ))}
          </div>

          {/* Current Step Info */}
          <div style={{ backgroundColor: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: '0.5rem', padding: '1rem' }}>
            <h3 style={{ fontWeight: '600', color: '#1e40af', margin: 0 }}>{PIPELINE_STEPS[currentStep - 1]?.name}</h3>
            <p style={{ color: '#1d4ed8', fontSize: '0.875rem', marginTop: '0.25rem', margin: 0 }}>{PIPELINE_STEPS[currentStep - 1]?.description}</p>
          </div>
        </div>

        {/* Step Content */}
        <div style={{ marginBottom: '2rem' }}>
          {renderStepContent()}
        </div>

        {/* Navigation Buttons */}
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <button
            onClick={goToPreviousStep}
            disabled={currentStep === 1 || isProcessing}
            style={{
              padding: '0.75rem 1.5rem',
              backgroundColor: '#f3f4f6',
              color: '#374151',
              borderRadius: '0.5rem',
              fontWeight: '500',
              border: 'none',
              cursor: (currentStep === 1 || isProcessing) ? 'not-allowed' : 'pointer',
              opacity: (currentStep === 1 || isProcessing) ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
          >
            이전 단계
          </button>

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            {/* 리셋 버튼 (처리 중이 아닐 때만) */}
            {!isProcessing && (currentStep > 1 || result) && (
              <button
                onClick={() => {
                  reset();
                  setCurrentStep(1);
                  setCompletedSteps([]);
                  setPersonImage(null);
                  setClothingImage(null);
                  setStepResults({});
                  setFileErrors({});
                }}
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: '#6b7280',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                처음부터
              </button>
            )}

            {currentStep < 8 && (
              <button
                onClick={processCurrentStep}
                disabled={!canProceedToNext() || isProcessing}
                style={{
                  padding: '0.75rem 1.5rem',
                  backgroundColor: (!canProceedToNext() || isProcessing) ? '#d1d5db' : '#3b82f6',
                  color: '#ffffff',
                  borderRadius: '0.5rem',
                  fontWeight: '500',
                  border: 'none',
                  cursor: (!canProceedToNext() || isProcessing) ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => {
                  const target = e.target as HTMLButtonElement;
                  if (!target.disabled) {
                    target.style.backgroundColor = '#2563eb';
                  }
                }}
                onMouseLeave={(e) => {
                  const target = e.target as HTMLButtonElement;
                  if (!target.disabled) {
                    target.style.backgroundColor = '#3b82f6';
                  }
                }}
              >
                {currentStep <= 2 ? '다음 단계' : 
                 currentStep === 7 ? '가상 피팅 시작' : 
                 isProcessing ? '처리 중...' : '처리 시작'}
              </button>
            )}
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div style={{ 
            marginTop: '1.5rem',
            backgroundColor: '#fef2f2', 
            border: '1px solid #fecaca', 
            borderRadius: '0.5rem', 
            padding: '1rem'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ display: 'flex' }}>
                <svg style={{ flexShrink: 0, height: '1.25rem', width: '1.25rem', color: '#f87171' }} viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div style={{ marginLeft: '0.75rem' }}>
                  <h3 style={{ fontSize: '0.875rem', fontWeight: '500', color: '#991b1b', margin: 0 }}>Error</h3>
                  <p style={{ fontSize: '0.875rem', color: '#b91c1c', marginTop: '0.25rem', margin: 0 }}>{error}</p>
                </div>
              </div>
              <button
                onClick={clearError}
                style={{
                  backgroundColor: 'transparent',
                  border: 'none',
                  color: '#991b1b',
                  cursor: 'pointer',
                  padding: '0.25rem'
                }}
              >
                ✕
              </button>
            </div>
          </div>
        )}

        {/* Instructions (첫 번째 단계에서만 표시) */}
        {currentStep === 1 && !personImage && !clothingImage && (
          <div style={{ 
            marginTop: '2rem',
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
            
            {/* 개발 도구 정보 */}
            <div style={{ 
              marginTop: '1.5rem', 
              padding: '1rem', 
              backgroundColor: '#f9fafb', 
              borderRadius: '0.5rem',
              fontSize: '0.875rem',
              color: '#4b5563'
            }}>
              <p style={{ margin: 0, fontWeight: '500' }}>🛠️ 개발자 도구:</p>
              <p style={{ margin: '0.25rem 0 0 0' }}>
                콘솔에서 <code>devTools.testAPI()</code>, <code>devTools.testDummyVirtualTryOn()</code> 등을 사용할 수 있습니다.
              </p>
            </div>
          </div>
        )}
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
        
        code {
          background-color: #f3f4f6;
          padding: 0.125rem 0.25rem;
          border-radius: 0.25rem;
          font-family: 'Courier New', monospace;
          font-size: 0.8em;
        }
      `}</style>
    </div>
  );
};

export default App;