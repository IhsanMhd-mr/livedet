import { Component } from 'react'

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    console.error('ErrorBoundary caught:', error, info)
  }

  reset = () => {
    this.setState({ hasError: false, error: null })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-surface-900 px-6 py-12">
          <div className="max-w-md rounded-2xl border border-danger-500/20 bg-danger-500/5 p-8 text-center">
            <div className="mb-4 flex justify-center">
              <div className="rounded-full bg-danger-500/10 p-3">
                <svg className="h-6 w-6 text-danger-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4v2m0 0v2m0-6v-2m0 0V7a2 2 0 012-2h2.586a1 1 0 00.707-.293l-2.414-2.414a1 1 0 00-1.414 1.414L10.586 7H8a1 1 0 000 2h2m0 0h2m-2 0v2m2-2v-2m0 0h2v2h-2z" />
                </svg>
              </div>
            </div>

            <h2 className="mb-2 text-lg font-bold text-white">Something went wrong</h2>
            <p className="mb-6 text-sm text-slate-400">An unexpected error occurred. Please try refreshing or reset.</p>

            <details className="mb-6 text-left">
              <summary className="cursor-pointer text-xs font-mono text-slate-300 hover:text-slate-200">Error details</summary>
              <pre className="mt-2 max-h-40 overflow-auto rounded bg-surface-800/60 p-2 text-xs text-slate-300">{this.state.error?.toString()}</pre>
            </details>

            <div className="flex gap-3">
              <button
                onClick={this.reset}
                className="flex-1 rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-600"
              >
                Try Again
              </button>
              <button
                onClick={() => window.location.reload()}
                className="flex-1 rounded-lg border border-white/10 px-4 py-2 text-sm font-semibold text-slate-300 hover:bg-white/5"
              >
                Reload
              </button>
            </div>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
