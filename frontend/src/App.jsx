import React, { useEffect, useMemo, useState, useCallback, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap, CircleMarker } from 'react-leaflet';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle2,
  XCircle,
  Navigation,
  Gauge,
  Search,
  Filter,
  RotateCcw,
  X,
  MapPin,
  Upload,
  Play,
  Loader2,
  FileSpreadsheet,
  Download,
  Globe,
  ShieldCheck,
  Folder,
  RefreshCw,
} from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// ============ API CONFIG ============
// Use current hostname for network access
const API_HOST = window.location.hostname || 'localhost';
const API_BASE = `http://${API_HOST}:8000`;
const WS_BASE = `ws://${API_HOST}:8000`;

// ============ COLOR SYSTEM ============
const COLOR = {
  emerald: {
    chip: 'bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-400/25',
    iconBg: 'bg-emerald-500/15',
    iconText: 'text-emerald-300',
    glow: 'shadow-[0_0_18px_rgba(16,185,129,0.22)]',
  },
  blue: {
    chip: 'bg-sky-500/15 text-sky-200 ring-1 ring-sky-400/25',
    iconBg: 'bg-sky-500/15',
    iconText: 'text-sky-300',
    glow: 'shadow-[0_0_18px_rgba(56,189,248,0.22)]',
  },
  purple: {
    chip: 'bg-purple-500/15 text-purple-200 ring-1 ring-purple-400/25',
    iconBg: 'bg-purple-500/15',
    iconText: 'text-purple-300',
    glow: 'shadow-[0_0_18px_rgba(168,85,247,0.22)]',
  },
  rose: {
    chip: 'bg-rose-500/15 text-rose-200 ring-1 ring-rose-400/25',
    iconBg: 'bg-rose-500/15',
    iconText: 'text-rose-300',
    glow: 'shadow-[0_0_18px_rgba(244,63,94,0.22)]',
  },
};

// ============ COMPONENTS ============

const LiquidCard = ({ children, className = "" }) => (
  <motion.div
    initial={{ opacity: 0, y: 10, scale: 0.98 }}
    animate={{ opacity: 1, y: 0, scale: 1 }}
    transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
    className={`liquid-surface rounded-3xl ${className}`}
  >
    {children}
  </motion.div>
);

const StatPill = ({ label, value, icon: Icon, tone = "blue" }) => {
  const t = COLOR[tone] || COLOR.blue;
  return (
    <motion.div
      whileHover={{ y: -2, scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      className={`liquid-mini rounded-2xl px-4 py-3 flex items-center gap-3 ${t.glow}`}
    >
      <div className={`h-10 w-10 rounded-2xl flex items-center justify-center ${t.iconBg} ring-1 ring-white/10`}>
        <Icon className={`h-5 w-5 ${t.iconText}`} />
      </div>
      <div className="min-w-0">
        <p className="text-[10px] text-white/55 uppercase tracking-[0.18em] font-semibold">{label}</p>
        <p className="text-xl leading-none font-extrabold text-white tracking-tight">{value}</p>
      </div>
    </motion.div>
  );
};

const FlyToLocation = ({ destination, zoom = 16 }) => {
  const map = useMap();
  useEffect(() => {
    if (destination && destination[0] != null && destination[1] != null) {
      map.flyTo(destination, zoom, { duration: 1.8, easeLinearity: 0.55 });
    }
  }, [destination, zoom, map]);
  return null;
};

const ZoomHUD = () => {
  const map = useMap();
  return (
    <div className="absolute right-5 top-5 z-[1200] flex flex-col gap-2">
      <button className="liquid-button w-11 h-11 rounded-2xl" onClick={() => map.zoomIn()} title="Zoom in">+</button>
      <button className="liquid-button w-11 h-11 rounded-2xl" onClick={() => map.zoomOut()} title="Zoom out">−</button>
    </div>
  );
};

const createGlassIcon = (status) => {
  let className = 'glass-marker';
  if (status === 'VALID') className += ' valid';
  else if (status === 'INVALID') className += ' invalid';
  else className += ' found'; // Neutral blue for just found

  return L.divIcon({
    className: 'custom-icon',
    html: `<div class="${className}"><div class="glass-marker__core"></div><div class="ripple-marker"></div></div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13],
    popupAnchor: [0, -13],
  });
};

const fmtMeters = (m) => {
  if (m == null || Number.isNaN(Number(m))) return "—";
  const n = Number(m);
  if (n >= 1000) return `${(n / 1000).toFixed(2)}km`;
  return `${Math.round(n)}m`;
};

// ============ UPLOAD MODAL ============
const UploadModal = ({ isOpen, onClose, onUpload, onCancel, scriptType, isLoading, progress, progressMessage }) => {
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && (dropped.name.endsWith('.xlsx') || dropped.name.endsWith('.xls'))) {
      setFile(dropped);
    }
  };

  const handleSubmit = () => {
    if (file) onUpload(file, scriptType);
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, y: 20 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 20 }}
        onClick={(e) => e.stopPropagation()}
        className="liquid-hud rounded-[32px] p-8 w-[500px] max-w-[90vw]"
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            {scriptType === 'validate' ? (
              <ShieldCheck className="h-6 w-6 text-emerald-400" />
            ) : (
              <Globe className="h-6 w-6 text-sky-400" />
            )}
            <h2 className="text-xl font-bold text-white">
              {scriptType === 'validate' ? 'Validar Coordenadas' : 'Buscar no Google Maps'}
            </h2>
          </div>
          <button onClick={onClose} className="liquid-button p-2 rounded-xl" disabled={isLoading}>
            <X className="h-5 w-5 text-white/70" />
          </button>
        </div>

        {!isLoading ? (
          <>
            <div
              className={`border-2 border-dashed rounded-2xl p-8 text-center transition-colors cursor-pointer
                ${dragOver ? 'border-sky-400 bg-sky-400/10' : 'border-white/20 hover:border-white/40'}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".xlsx,.xls"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              <FileSpreadsheet className="h-12 w-12 text-white/40 mx-auto mb-4" />
              {file ? (
                <p className="text-white font-semibold">{file.name}</p>
              ) : (
                <>
                  <p className="text-white/60">Arraste a planilha Excel aqui</p>
                  <p className="text-white/40 text-sm mt-1">ou clique para selecionar</p>
                </>
              )}
            </div>

            <button
              onClick={handleSubmit}
              disabled={!file}
              className={`mt-6 w-full liquid-button rounded-2xl py-4 flex items-center justify-center gap-3 text-lg font-bold
                ${file ? 'text-white' : 'text-white/40 cursor-not-allowed'}`}
            >
              <Play className="h-5 w-5" />
              Executar
            </button>
          </>
        ) : (
          <div className="py-8">
            <div className="flex items-center justify-center gap-3 mb-4">
              <Loader2 className="h-6 w-6 text-sky-400 animate-spin" />
              <span className="text-white font-semibold">Processando...</span>
            </div>
            <div className="h-3 rounded-full bg-white/10 overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-sky-400 to-emerald-400"
                initial={{ width: 0 }}
                animate={{ width: `${Math.max(0, progress)}%` }}
                transition={{ duration: 0.3 }}
              />
            </div>
            <p className="text-white/60 text-sm text-center mt-3">{progressMessage || `${progress}%`}</p>

            {/* Cancel Button */}
            <button
              onClick={onCancel}
              className="mt-6 w-full liquid-button rounded-2xl py-3 flex items-center justify-center gap-2 text-rose-400 hover:text-rose-300"
            >
              <X className="h-4 w-4" />
              Cancelar
            </button>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
};

// ============ FILE CENTER MODAL ============
const FileCenterModal = ({ isOpen, onClose, files }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="liquid-surface rounded-3xl w-full max-w-2xl max-h-[80vh] flex flex-col overflow-hidden border border-white/20 bg-[#121420]"
      >
        <div className="p-6 border-b border-white/10 flex justify-between items-center">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-xl bg-purple-500/20 text-purple-300`}>
              <Folder className="h-5 w-5" />
            </div>
            <h2 className="text-xl font-bold text-white">Central de Arquivos</h2>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-full transition"><X className="h-5 w-5 text-white/70" /></button>
        </div>
        <div className="flex-1 overflow-y-auto p-6 space-y-3 custom-scrollbar">
          {files.length === 0 ? (
            <div className="text-center text-white/40 py-10">Nenhum arquivo encontrado.</div>
          ) : files.map((f, i) => (
            <div key={i} className="flex items-center justify-between p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition group">
              <div className="flex items-center gap-3 overflow-hidden">
                <FileSpreadsheet className="h-8 w-8 text-emerald-400 opacity-80" />
                <div className="min-w-0">
                  <p className="font-semibold text-white/90 truncate">{f.name}</p>
                  <p className="text-xs text-white/50">{new Date(f.modified * 1000).toLocaleString()} • {(f.size / 1024).toFixed(1)} KB</p>
                </div>
              </div>
              <a
                href={`${API_BASE}${f.url}`}
                className="liquid-button px-4 py-2 rounded-lg text-xs font-bold flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity"
                download
              >
                <Download className="h-3 w-3" /> Baixar
              </a>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

// ============ MAIN APP ============
function App() {
  const [data, setData] = useState([]);
  const [selectedRow, setSelectedRow] = useState(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("ALL");
  const [activeTab, setActiveTab] = useState("validator"); // validator | search

  // Upload state
  const [showUpload, setShowUpload] = useState(false);
  const [uploadScript, setUploadScript] = useState('validate');
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState("");
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);
  const wsRef = useRef(null);

  // File Center state
  const [showFiles, setShowFiles] = useState(false);
  const [fileList, setFileList] = useState([]);

  useEffect(() => {
    console.log("[DEBUG] State Update:", { activeTab, rows: data.length, jobId: currentJobId });
  }, [activeTab, data.length, currentJobId]);

  // Load initial data from static file (if exists)
  useEffect(() => {
    fetch('/results.json')
      .then(res => res.ok ? res.json() : [])
      .then(setData)
      .catch(() => setData([]));
  }, []);

  const stats = useMemo(() => {
    const total = data.length;
    const valid = data.filter(d => d.VALIDATION_STATUS === 'VALID').length;
    const invalid = data.filter(d => d.VALIDATION_STATUS === 'INVALID').length;
    const avg = total > 0
      ? (data.reduce((acc, curr) => acc + (Number(curr.MATCH_SCORE) || Number(curr.Validation_Score) || 0), 0) / total).toFixed(1)
      : "0.0";
    return { total, valid, invalid, avg };
  }, [data]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data
      .filter(r => {
        if (filter === "VALID") return r.VALIDATION_STATUS === "VALID";
        if (filter === "INVALID") return r.VALIDATION_STATUS !== "VALID";
        return true;
      })
      .filter(r => {
        if (!q) return true;
        const s = `${r.COMPLETO || ""} ${r.DS_STREET || ""} ${r.NOMINATIM_ADDRESS || ""}`.toLowerCase();
        return s.includes(q);
      });
  }, [data, query, filter]);

  const reset = useCallback(() => setSelectedRow(null), []);



  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'r' || e.key === 'R') reset();
      if (e.key === 'Escape') reset();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [reset]);

  // Reusable WebSocket connection function
  const connectWebSocket = useCallback((job_id) => {
    const ws = new WebSocket(`${WS_BASE}/ws/${job_id}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data);
        console.log('WS message:', msg);

        if (msg.type === 'progress') {
          setProgress(msg.progress);
          setProgressMessage(msg.message);
        } else if (msg.type === 'completed') {
          if (msg.results_url) {
            setProgress(100);
            setProgressMessage("Baixando resultados...");
            try {
              const url = `${API_BASE}${msg.results_url}`;
              console.log("Fetching results from:", url);
              const res = await fetch(url);
              if (!res.ok) {
                const text = await res.text();
                console.error("Fetch failed:", res.status, text);
                throw new Error(`Server returned ${res.status}`);
              }
              const jsonParams = await res.json();
              setData(jsonParams || []);
            } catch (e) {
              console.error("Error fetching results JSON:", e);
              alert(`Erro ao baixar resultados: ${e.message}`);
            }
          } else {
            setData(msg.results || []);
          }

          setDownloadUrl(msg.download_url);
          setIsLoading(false);
          setShowUpload(false);
          setProgress(100);
          wsRef.current = null;
        } else if (msg.type === 'error') {
          alert(`Erro: ${msg.message}`);
          setIsLoading(false);
          wsRef.current = null;
        }
      } catch (e) {
        console.error('WS parse error:', e);
      }
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event.code, event.reason);
      wsRef.current = null;
    };

    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      // alert('Erro de conexão WebSocket. Verifique se o backend está rodando.');
      wsRef.current = null;
    };
  }, []);

  const handleUpload = async (file, scriptType) => {
    setIsLoading(true);
    setProgress(0);
    setProgressMessage("Enviando arquivo...");
    setDownloadUrl(null);

    try {
      // 1. Upload file and get job ID
      const formData = new FormData();
      formData.append('file', file);

      const uploadRes = await fetch(`${API_BASE}/run/${scriptType}`, {
        method: 'POST',
        body: formData,
      });

      if (!uploadRes.ok) throw new Error('Upload failed');
      const { job_id } = await uploadRes.json();
      setCurrentJobId(job_id);
      setProgressMessage("Conectando ao servidor...");

      // 2. Connect WebSocket for progress
      connectWebSocket(job_id);

    } catch (err) {
      alert(`Erro: ${err.message}`);
      setIsLoading(false);
    }
  };

  const openUploadModal = (script) => {
    setUploadScript(script);
    setShowUpload(true);
  };

  const handleCancel = () => {
    // Signal backend to cancel
    if (currentJobId) {
      fetch(`${API_BASE}/job/${currentJobId}/cancel`, { method: 'POST' })
        .catch(err => console.error("Error sending cancel signal:", err));
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsLoading(false);
    setProgress(0);
    setProgressMessage('Operação cancelada.');
  };

  const handleValidateResults = async () => {
    if (!currentJobId) {
      alert("Nenhum job recente para validar.");
      return;
    }
    setIsLoading(true);
    setShowUpload(true);
    setProgress(0);
    setProgressMessage("Iniciando validação detalhada (Nominatim + DeepSeek)...");

    try {
      const res = await fetch(`${API_BASE}/job/${currentJobId}/validate`, { method: 'POST' });
      if (!res.ok) throw new Error("Erro ao iniciar validação");

      const { new_job_id } = await res.json();
      console.log("New Validation Job started:", new_job_id);

      setCurrentJobId(new_job_id);

      // Reconnect WebSocket to track the NEW job
      if (wsRef.current) {
        wsRef.current.close();
      }
      setTimeout(() => connectWebSocket(new_job_id), 500);

    } catch (e) {
      alert(`Erro: ${e.message}`);
      setIsLoading(false);
    }
  };

  const openFileCenter = async () => {
    try {
      const res = await fetch(`${API_BASE}/files`);
      if (!res.ok) throw new Error("Failed to fetch files");
      const files = await res.json();
      setFileList(files);
      setShowFiles(true);
    } catch (e) {
      alert(`Error fetching files: ${e.message}`);
    }
  };

  const handleRetryInvalid = async () => {
    const invalidRows = data.filter(r => r.VALIDATION_STATUS === 'INVALID');
    if (invalidRows.length === 0) {
      alert("Nenhuma linha inválida para reprocessar.");
      return;
    }

    setIsLoading(true);
    setShowUpload(true); // Show upload modal to display progress
    setProgress(0);
    setProgressMessage("Reprocessando linhas inválidas...");

    try {
      const res = await fetch(`${API_BASE}/job/${currentJobId}/retry_invalid`, {
        method: 'POST',
      });

      if (!res.ok) throw new Error("Erro ao iniciar reprocessamento de inválidos");

      const { new_job_id } = await res.json();
      setCurrentJobId(new_job_id);

      if (wsRef.current) {
        wsRef.current.close();
      }
      setTimeout(() => connectWebSocket(new_job_id), 500);

    } catch (e) {
      alert(`Erro ao reprocessar inválidos: ${e.message}`);
      setIsLoading(false);
    }
  };

  const handleRetryPending = async () => {
    // Pending logic: no coords found (Latitude/Longitude is null)
    const pendingRows = data.filter(r => {
      const lat = r.LAT || r.Latitude;
      const lng = r.LONG || r.Longitude;
      return lat == null || lng == null;
    });

    if (pendingRows.length === 0) {
      alert("Nenhuma linha pendente para reprocessar.");
      return;
    }

    setIsLoading(true);
    setShowUpload(true);
    setProgress(0);
    setProgressMessage("Reprocessando linhas pendentes...");

    try {
      const res = await fetch(`${API_BASE}/job/${currentJobId}/retry_pending`, {
        method: 'POST'
      });

      if (!res.ok) throw new Error("Erro ao iniciar reprocessamento de pendentes");

      const { new_job_id } = await res.json();
      setCurrentJobId(new_job_id);

      if (wsRef.current) {
        wsRef.current.close();
      }
      setTimeout(() => connectWebSocket(new_job_id), 500);
    } catch (e) {
      alert(`Erro ao reprocessar pendentes: ${e.message}`);
      setIsLoading(false);
    }
  };

  return (
    <div
      className="relative min-h-screen w-full overflow-hidden bg-black selection:bg-pink-500/30"
      style={{ backgroundColor: '#000', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}
    >
      {/* Background */}
      <div className="absolute inset-0 z-0 pointer-events-none">
        <div className="absolute inset-0 noise-layer opacity-[0.18]" />
        <div className="absolute -top-28 -left-24 w-[34rem] h-[34rem] bg-purple-500/40 rounded-full blur-3xl mix-blend-screen animate-blob" />
        <div className="absolute -top-24 -right-24 w-[34rem] h-[34rem] bg-sky-500/35 rounded-full blur-3xl mix-blend-screen animate-blob animation-delay-2000" />
        <div className="absolute -bottom-24 left-24 w-[34rem] h-[34rem] bg-pink-500/35 rounded-full blur-3xl mix-blend-screen animate-blob animation-delay-4000" />
      </div>

      {/* Layout */}
      <div className="relative z-10 h-screen w-full px-6 py-6 flex flex-col">

        {/* Top Toolbar */}
        <div className="flex items-center justify-between mb-6 shrink-0">
          <div className="flex items-center gap-4">
            <h1 className="text-2xl font-black text-white tracking-tight">GeoValidator</h1>

            {/* Tab Switcher */}
            <div className="flex items-center gap-1 p-1 rounded-2xl bg-white/5 ring-1 ring-white/10">
              <button
                onClick={() => setActiveTab('validator')}
                className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all flex items-center gap-2
                  ${activeTab === 'validator' ? 'bg-white/10 text-white' : 'text-white/50 hover:text-white/70'}`}
              >
                <ShieldCheck className="h-4 w-4" />
                Validador
              </button>
              <button
                onClick={() => setActiveTab('search')}
                className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all flex items-center gap-2
                  ${activeTab === 'search' ? 'bg-white/10 text-white' : 'text-white/50 hover:text-white/70'}`}
              >
                <Globe className="h-4 w-4" />
                Busca Maps
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => openUploadModal('validate')}
              className="liquid-button rounded-2xl px-4 py-2.5 flex items-center gap-2"
            >
              <Upload className="h-4 w-4 text-emerald-400" />
              <span className="text-sm font-semibold text-white/90">Validar Planilha</span>
            </button>
            <button
              onClick={() => openUploadModal('search')}
              className="liquid-button rounded-2xl px-4 py-2.5 flex items-center gap-2"
            >
              <Upload className="h-4 w-4 text-sky-400" />
              <span className="text-sm font-semibold text-white/90">Buscar Maps</span>
            </button>
            <a
              href={`${API_BASE}/download/template`}
              className="liquid-button rounded-2xl px-4 py-2.5 flex items-center gap-2"
              download
            >
              <FileSpreadsheet className="h-4 w-4 text-purple-400" />
              <span className="text-sm font-semibold text-white/90">Modelo</span>
            </a>
            {activeTab === 'search' && data.length > 0 && currentJobId && (
              <button
                onClick={handleValidateResults}
                className="liquid-button rounded-2xl px-4 py-2.5 flex items-center gap-2"
                disabled={isLoading}
              >
                <ShieldCheck className="h-4 w-4 text-emerald-400" />
                <span className="text-sm font-semibold text-white/90">Validar Encontrados</span>
              </button>
            )}
            {downloadUrl && (
              <a
                href={`${API_BASE}${downloadUrl}`}
                className="liquid-button rounded-2xl px-4 py-2.5 flex items-center gap-2"
              >
                <Download className="h-4 w-4 text-purple-400" />
                <span className="text-sm font-semibold text-white/90">Baixar Resultado</span>
              </a>
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-12 gap-6 flex-1 min-h-0">
          {/* Left panel */}
          <div className="col-span-4 flex flex-col gap-6 h-full min-h-0">
            <LiquidCard className="p-6 liquid-hover shrink-0">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm text-white/55 font-light">
                    {activeTab === 'validator' ? 'Validação de Coordenadas' : 'Busca Google Maps'}
                  </p>
                </div>
                <div className="flex gap-2">
                  <button className="liquid-button rounded-2xl px-3 py-2 flex items-center gap-2" onClick={openFileCenter} title="Histórico de Arquivos">
                    <Folder className="h-4 w-4 text-purple-300" />
                  </button>
                  <button className="liquid-button rounded-2xl px-3 py-2 flex items-center gap-2" onClick={reset} title="Reset (R)">
                    <RotateCcw className="h-4 w-4 text-white/80" />
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mt-5">
                <StatPill label="Valid" value={stats.valid} icon={CheckCircle2} tone="emerald" />
                <StatPill label="Total" value={stats.total} icon={Navigation} tone="blue" />
                <StatPill label="Score" value={stats.avg} icon={Gauge} tone="purple" />
                <StatPill label="Invalid" value={stats.invalid} icon={XCircle} tone="rose" />
              </div>
            </LiquidCard>

            <LiquidCard className="flex-1 basis-0 overflow-hidden flex flex-col p-0 min-h-0">
              <div className="sticky top-0 z-20 px-5 pt-5 pb-4 border-b border-white/10 shrink-0">
                <div className="flex items-center justify-between gap-3">
                  <h2 className="text-lg font-bold text-white">Resultados</h2>
                  <div className="flex gap-2">
                    {["ALL", "VALID", "INVALID"].map((k) => (
                      <button
                        key={k}
                        onClick={() => setFilter(k)}
                        className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition
                          ${filter === k ? "bg-white/12 ring-1 ring-white/20 text-white" : "bg-white/5 text-white/50"}`}
                      >
                        {k}
                      </button>
                    ))}
                  </div>

                  {/* Retry Button (Only if has invalid and job completed/results loaded) */}
                  {stats.invalid > 0 && (
                    <button
                      onClick={handleRetryInvalid}
                      className="ml-2 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 px-3 py-1.5 rounded-xl text-xs font-bold flex items-center gap-2 border border-rose-500/30 transition shadow-[0_0_12px_rgba(244,63,94,0.2)]"
                      title="Reprocessar linhas inválidas no Google Maps + Validação"
                    >
                      <RefreshCw className="h-3 w-3" /> Retry Invalid
                    </button>
                  )}
                  {/* Retry Pending Button */}
                  {data.some(r => (r.LAT || r.Latitude) == null) && (
                    <button
                      onClick={handleRetryPending}
                      className="ml-2 bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 px-3 py-1.5 rounded-xl text-xs font-bold flex items-center gap-2 border border-sky-500/30 transition shadow-[0_0_12px_rgba(56,189,248,0.2)]"
                      title="Retentar buscar linhas sem coordenadas encontradas"
                    >
                      <RefreshCw className="h-3 w-3" /> Retry Pending
                    </button>
                  )}
                </div>
                <div className="mt-3 relative">
                  <Search className="h-4 w-4 text-white/45 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Buscar..."
                    className="w-full liquid-input rounded-xl pl-10 pr-4 py-2.5 text-sm text-white placeholder:text-white/35"
                  />
                </div>
              </div>

              <div className="overflow-y-auto p-4 space-y-3 flex-1 custom-scrollbar">
                {data.length === 0 ? (
                  <div className="text-center py-16 text-white/40">
                    <FileSpreadsheet className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>Nenhum dado carregado</p>
                    <p className="text-sm mt-1">Use os botões acima para processar uma planilha</p>
                  </div>
                ) : filtered.map((row, idx) => {
                  const active = selectedRow === row;
                  const isValid = row.VALIDATION_STATUS === "VALID";
                  const isInvalid = row.VALIDATION_STATUS === "INVALID";
                  const status = row.VALIDATION_STATUS;
                  const score = Number(row.MATCH_SCORE) || Number(row.Validation_Score) || 0;

                  // Normalize coordinates
                  const lat = row.LAT || row.Latitude;
                  const lng = row.LONG || row.Longitude;
                  const hasCoords = lat != null && lng != null;

                  return (
                    <motion.button
                      key={`${idx}-${lat}-${lng}`}
                      whileHover={{ y: -2 }}
                      whileTap={{ scale: 0.99 }}
                      onClick={() => setSelectedRow(row)}
                      className={`w-full text-left rounded-2xl p-4 transition relative overflow-hidden
                        ${active ? "bg-white/12 ring-1 ring-sky-300/30" : "bg-white/5 ring-1 ring-white/10 hover:bg-white/8"}`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        {status ? (
                          <span className={`px-2 py-1 rounded-full text-[10px] font-bold uppercase
                            ${isValid ? "bg-emerald-500/20 text-emerald-200" : "bg-rose-500/20 text-rose-200"}`}>
                            {status}
                          </span>
                        ) : hasCoords ? (
                          <span className="px-2 py-1 rounded-full text-[10px] font-bold uppercase bg-sky-500/20 text-sky-200 flex items-center gap-1">
                            <MapPin className="h-3 w-3" /> Found
                          </span>
                        ) : (
                          <span className="bg-white/10 text-white/40 px-2 py-1 rounded-full text-[10px] font-bold uppercase">Pending</span>
                        )}
                        <span className="text-xs text-white/50">{fmtMeters(row.DISTANCE_METERS)}</span>
                      </div>
                      <h3 className="text-sm font-semibold text-white/95 truncate mt-2">{row.COMPLETO || row.Maps_Title}</h3>
                      {hasCoords && !status && (
                        <div className="mt-1 flex items-center gap-2">
                          <p className="text-[10px] text-white/40 font-mono">{Number(lat).toFixed(5)}, {Number(lng).toFixed(5)}</p>
                        </div>
                      )}

                      <div className="mt-2 h-1.5 rounded-full bg-white/10 overflow-hidden">
                        <div className={`h-full rounded-full ${isValid ? 'bg-emerald-400' : 'bg-sky-400/70'}`} style={{ width: `${Math.min(100, score * 100 || score)}%` }} />
                      </div>
                    </motion.button>
                  );
                })}
              </div>
            </LiquidCard>
          </div>

          {/* Right panel (map) */}
          <div className="col-span-8 h-full relative rounded-[32px] overflow-hidden border border-white/15 shadow-2xl bg-[#0f1014]">
            <div className="absolute inset-0 z-0">
              <MapContainer center={[-16.68, -49.25]} zoom={13} style={{ height: '100%', width: '100%' }} zoomControl={false} preferCanvas={true}>
                <TileLayer attribution='&copy; OpenStreetMap' url="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png" />
                <ZoomHUD />
                <FlyToLocation destination={
                  (() => {
                    if (!selectedRow) return null;
                    const lat = selectedRow.LAT || selectedRow.Latitude;
                    const lng = selectedRow.LONG || selectedRow.Longitude;
                    return (lat != null && lng != null) ? [lat, lng] : null;
                  })()
                } zoom={16} />
                {data.map((row, idx) => {
                  const lat = row.LAT || row.Latitude;
                  const lng = row.LONG || row.Longitude;
                  const hasCoords = lat != null && lng != null;
                  if (!hasCoords) return null;

                  const fillColor = row.VALIDATION_STATUS === 'VALID' ? '#10b981' : (row.VALIDATION_STATUS === 'INVALID' ? '#f43f5e' : '#38bdf8');
                  const isSelected = selectedRow === row;

                  return (
                    <CircleMarker
                      key={idx}
                      center={[lat, lng]}
                      radius={isSelected ? 8 : 4}
                      pathOptions={{
                        color: isSelected ? '#fff' : fillColor,
                        fillColor: fillColor,
                        fillOpacity: isSelected ? 1 : 0.7,
                        weight: isSelected ? 2 : 0
                      }}
                      eventHandlers={{ click: () => setSelectedRow(row) }}
                    >
                      {isSelected && (
                        <Popup className="glass-popup">
                          <div className="glass-popup-inner">
                            <div className="flex items-center gap-2">
                              <MapPin className="h-4 w-4 text-slate-500" />
                              {row.VALIDATION_STATUS ? (
                                <span className={`text-xs font-bold uppercase ${row.VALIDATION_STATUS === 'VALID' ? 'text-emerald-600' : 'text-rose-600'}`}>
                                  {row.VALIDATION_STATUS}
                                </span>
                              ) : (
                                <span className="text-xs font-bold uppercase text-sky-600">FOUND</span>
                              )}
                            </div>
                            <div className="mt-2 text-sm text-slate-600 font-extrabold">{row.NOMINATIM_ADDRESS?.split(',')[0] || row.Maps_Title?.split(',')[0] || "Local Encontrado"}</div>
                            <div className="mt-1 text-xs text-slate-500 font-medium">
                              {row.DISTANCE_METERS ? <>Dist: <span className="text-slate-600 font-bold">{fmtMeters(row.DISTANCE_METERS)}</span> • </> : null}
                              Score: <span className="text-slate-600 font-bold">{row.MATCH_SCORE || row.Validation_Score || "—"}</span>
                            </div>
                          </div>
                        </Popup>
                      )}
                    </CircleMarker>
                  );
                })}
              </MapContainer>
            </div>

            {/* HUD */}
            <AnimatePresence>
              {selectedRow && (
                <motion.div
                  initial={{ opacity: 0, y: -18 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="absolute left-6 right-6 top-6 z-[1300] pointer-events-none"
                >
                  <div className="liquid-hud rounded-[24px] px-5 py-4 pointer-events-auto">
                    <div className="flex items-center justify-between gap-4">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`px-2 py-1 rounded-full text-[10px] font-bold uppercase
                            ${selectedRow.VALIDATION_STATUS === 'VALID' ? "bg-emerald-500/20 text-emerald-200" : "bg-rose-500/20 text-rose-200"}`}>
                            {selectedRow.VALIDATION_STATUS}
                          </span>
                          <span className="text-xs text-white/50">Score: {selectedRow.MATCH_SCORE}</span>
                        </div>
                        <h3 className="text-sm font-semibold text-white truncate">{selectedRow.COMPLETO}</h3>
                        <p className="text-xs text-white/50 truncate mt-1">{selectedRow.NOMINATIM_ADDRESS}</p>
                      </div>
                      <button onClick={reset} className="liquid-button p-2 rounded-xl shrink-0">
                        <X className="h-4 w-4 text-white/70" />
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Legend */}
            <div className="absolute right-6 bottom-6 z-[1300] flex gap-2 pointer-events-none">
              <div className="liquid-chip flex items-center gap-2 pointer-events-auto">
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_14px_rgba(16,185,129,0.7)]" />
                VALID
              </div>
              <div className="liquid-chip flex items-center gap-2 pointer-events-auto">
                <span className="h-2.5 w-2.5 rounded-full bg-rose-400 shadow-[0_0_14px_rgba(244,63,94,0.7)]" />
                INVALID
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Modal */}
      <AnimatePresence>
        {showUpload && (
          <UploadModal
            isOpen={showUpload}
            onClose={() => !isLoading && setShowUpload(false)}
            onUpload={handleUpload}
            onCancel={handleCancel}
            scriptType={uploadScript}
            isLoading={isLoading}
            progress={progress}
            progressMessage={progressMessage}
          />
        )}
      </AnimatePresence>
      {/* File Center Modal */}
      <AnimatePresence>
        {showFiles && <FileCenterModal isOpen={showFiles} onClose={() => setShowFiles(false)} files={fileList} />}
      </AnimatePresence>
    </div >
  );
};

export default App;