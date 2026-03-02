import axios from 'axios'
import type { PredictResponse, HealthResponse } from '@/types/detection'

// ─── Base Client ──────────────────────────────────────────────────────────────

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
  timeout: 60_000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg =
      err.response?.data?.error ??
      err.response?.data?.message ??
      err.message ??
      'Unknown error'
    return Promise.reject(new Error(msg))
  },
)

// ─── Endpoints ────────────────────────────────────────────────────────────────

export async function predictImage(
  file: File,
  confidence?: number,
): Promise<PredictResponse> {
  const form = new FormData()
  form.append('image', file)
  if (confidence !== undefined) form.append('confidence', String(confidence))

  const { data } = await api.post<PredictResponse>('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health')
  return data
}

export default api
