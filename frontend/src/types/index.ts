// 기본 타입 정의들
export interface UserMeasurements {
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hips?: number;
}

export interface VirtualTryOnRequest {
  personImage: File;
  clothingImage: File;
  height: number;
  weight: number;
  chest?: number;
  waist?: number;
  hips?: number;
}

export interface VirtualTryOnResponse {
  task_id: string;
  status: 'processing' | 'completed' | 'error';
  message: string;
  estimated_time?: string;
}

export interface TaskStatus {
  status: 'processing' | 'completed' | 'error';
  progress: number;
  current_step: string;
  steps: Array<{
    id: string;
    name: string;
    status: 'pending' | 'processing' | 'completed' | 'error';
  }>;
  result?: TryOnResult;
  error?: string;
  created_at: number;
  completed_at?: number;
}

export interface TryOnResult {
  fitted_image: string;
  confidence: number;
  processing_time: number;
  body_analysis: {
    measurements: Record<string, number>;
    pose_keypoints: number[][];
    body_type: string;
  };
  clothing_analysis: {
    category: string;
    style: string;
    colors: string[];
    pattern: string;
  };
  fit_score: number;
  recommendations: string[];
  model_used: string;
  image_specs: {
    resolution: [number, number];
    format: string;
    quality: number;
  };
}
