// api/client.ts
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const virtualTryOnAPI = {
  async process(
    personImage: File,
    clothingImage: File,
    height: number,
    weight: number
  ): Promise<TryOnResult> {
    const formData = new FormData();
    formData.append('person_image', personImage);
    formData.append('clothing_image', clothingImage);
    formData.append('height', height.toString());
    formData.append('weight', weight.toString());

    const response = await fetch(`${API_BASE_URL}/api/v1/virtual-tryon`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Processing failed');
    }

    return response.json();
  },

  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/health`);
      const data = await response.json();
      return data.status === 'healthy';
    } catch {
      return false;
    }
  }
};