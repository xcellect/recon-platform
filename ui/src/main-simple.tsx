import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import ReCoNApp from './ReCoNApp.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ReCoNApp />
  </StrictMode>,
)