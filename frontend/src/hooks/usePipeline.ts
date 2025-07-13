import { useState, useCallback } from 'react';
import { apiClient, VirtualTryOnRequest, VirtualTryOnResponse, PipelineProgress } from '../services/api';

export const usePipeline = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState<PipelineProgress | null>(null);
  const [result, setResult] = useState<VirtualTryOnResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const processVirtualTryOn = useCallback(async (request: VirtualTryOnRequest) => {
    setIsProcessing(true);
    setError(null);
    setResult(null);
    setProgress(null);

    try {
      const response = await apiClient.processVirtualTryOn(
        request,
        (progressData) => setProgress(progressData)
      );
      
      setResult(response);
      return response;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const reset = useCallback(() => {
    setIsProcessing(false);
    setProgress(null);
    setResult(null);
    setError(null);
  }, []);

  return {
    isProcessing,
    progress,
    result,
    error,
    processVirtualTryOn,
    reset
  };
};  