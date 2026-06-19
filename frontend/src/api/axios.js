import axios from 'axios'
import toast from 'react-hot-toast'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 60000,
  withCredentials: false,  // Don't send cookies with requests to avoid header size issues
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.config?.silent) return Promise.reject(error)

    if (error.response?.status === 413) {
      toast.error('File too large')
    } else if (error.response?.status === 431) {
      toast.error('Request headers too large — try uploading a smaller file')
    } else if (error.response?.status === 422) {
      toast.error('Invalid file format')
    } else if (error.response?.status >= 500) {
      toast.error('Server error — try again later')
    } else if (!error.response) {
      toast.error('Backend unreachable')
    }

    return Promise.reject(error)
  },
)

export default api
