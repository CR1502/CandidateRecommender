import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MotionConfig } from 'framer-motion'
import { Masthead } from './components/layout/Masthead'
import Home from './pages/Home'
import Results from './pages/Results'

const queryClient = new QueryClient()

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* Respect the OS "reduce motion" setting for every animation */}
      <MotionConfig reducedMotion="user">
        <BrowserRouter>
          <Masthead />
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </BrowserRouter>
      </MotionConfig>
    </QueryClientProvider>
  )
}
