#!/bin/bash

# ============================================
# GeoValidator Dashboard - Script de Inicialização
# ============================================
# Este script inicia o backend (FastAPI) e o frontend (React/Vite)
# 
# USO: ./run.sh
# ============================================

echo "🚀 Iniciando GeoValidator Dashboard..."
echo ""

# Diretório base (onde está o script)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# Cores para output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para matar processos ao sair
cleanup() {
    echo ""
    echo -e "${YELLOW}⏹️  Encerrando servidores...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============ Verificar dependências ============
echo -e "${BLUE}📦 Verificando dependências...${NC}"

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 não encontrado. Instale com: brew install python3"
    exit 1
fi

# Verificar Node
if ! command -v npm &> /dev/null; then
    echo "❌ NPM não encontrado. Instale Node.js: brew install node"
    exit 1
fi

# Verificar dependências Python (silencioso)
pip3 install -q fastapi uvicorn python-multipart pandas openpyxl requests geopy thefuzz playwright 2>/dev/null

# Verificar se Playwright está instalado
if ! python3 -c "from playwright.async_api import async_playwright" 2>/dev/null; then
    echo -e "${YELLOW}📥 Instalando Playwright Chromium...${NC}"
    playwright install chromium
fi

# Verificar dependências frontend
if [ ! -d "$BASE_DIR/frontend/node_modules" ]; then
    echo -e "${YELLOW}📥 Instalando dependências do frontend...${NC}"
    cd "$BASE_DIR/frontend"
    npm install
    cd "$BASE_DIR"
fi

echo -e "${GREEN}✅ Dependências OK${NC}"
echo ""

# ============ Iniciar Backend ============
echo -e "${BLUE}🔧 Iniciando Backend (FastAPI) na porta 8000...${NC}"
cd "$BASE_DIR"
python3 -c "import uvicorn; uvicorn.run('api:app', host='0.0.0.0', port=8000)" &
BACKEND_PID=$!
sleep 2

# Verificar se backend iniciou
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Falha ao iniciar o backend"
    exit 1
fi

echo -e "${GREEN}✅ Backend rodando em http://localhost:8000${NC}"
echo ""

# ============ Iniciar Frontend ============
echo -e "${BLUE}🎨 Iniciando Frontend (Vite) na porta 3003...${NC}"
cd "$BASE_DIR/frontend"
npm run dev -- --port 3003 --host &
FRONTEND_PID=$!
sleep 3

# Verificar se frontend iniciou
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Falha ao iniciar o frontend"
    kill $BACKEND_PID
    exit 1
fi

echo -e "${GREEN}✅ Frontend rodando em http://localhost:3003${NC}"
echo ""

# ============ Exibir informações ============
echo "============================================"
echo -e "${GREEN}🎉 GeoValidator Dashboard está rodando!${NC}"
echo "============================================"
echo ""
echo -e "  📊 Dashboard:  ${BLUE}http://localhost:3003${NC}"
echo -e "  🔌 API:        ${BLUE}http://localhost:8000${NC}"
echo ""

# Obter IP da rede local
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "N/A")
if [ "$LOCAL_IP" != "N/A" ]; then
    echo -e "  🌐 Rede Local: ${BLUE}http://$LOCAL_IP:3003${NC}"
    echo ""
fi

echo "  Pressione Ctrl+C para encerrar"
echo "============================================"
echo ""

# Manter script rodando
wait
