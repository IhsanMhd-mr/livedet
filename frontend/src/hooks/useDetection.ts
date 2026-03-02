import { useState, useCallback } from 'react'
import { predictImage } from '@/api/axios'
import type { PredictResponse } from '@/types/detection'

interface State {
  loading: boolean
  result: PredictResponse | null
  error: string | null
}

export function useDetection() {
  const [state, setState] = useState<State>({
    loading: false,
    result: null,
    error: null,
  })

  const detect = useCallback(async (file: File, confidence?: number) => {
    setState({ loading: true, result: null, error: null })
    try {
      const result = await predictImage(file, confidence)
      setState({ loading: false, result, error: null })
      return result
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setState({ loading: false, result: null, error: msg })
      return null
    }
  }, [])

  const reset = useCallback(() => {
    setState({ loading: false, result: null, error: null })
  }, [])

  return { ...state, detect, reset }
}
