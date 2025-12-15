# GeoValidator & MapSearch 🌍📍

**GeoValidator** é uma solução completa para enriquecimento, geolocalização e validação de endereços. Utilizando automação de browser (Playwright) e Inteligência Artificial (DeepSeek), o sistema é capaz de encontrar coordenadas precisas no Google Maps e validar semanticamente se o resultado corresponde ao local desejado.

## ✨ Principais Funcionalidades

### 🔍 Busca Inteligente (Geo-Search)
- **Estratégia em Cascata**: Tenta buscar pelo "Nome Completo + Endereço". Se a confiança for baixa, tenta automaticamente estratégias alternativas ("Apenas Endereço" ou "Nome + Cidade").
- **Google Maps Automation**: Navega, clica e extrai dados reais (Pin !3d ou Viewport @lat,lon) simulando um usuário real para máxima precisão.

### 🛡️ Validação com IA (Deep Validation)
- **DeepSeek Integration**: Utiliza LLM (Large Language Model) para "ler" e comparar o endereço de entrada com o resultado encontrado.
- **Detecção de Falsos Positivos**: Identifica se o Google retornou um centro de cidade genérico em vez do endereço específico.
- **Score de Confiança**: Atribui uma nota (0-100%) para cada resultado.

### 🔄 Smart Retry & Merge
- **Reprocessamento Seletivo**: Permite reprocessar apenas linhas inválidas ou pendentes sem perder o trabalho já feito.
- **Merge Automático**: Mescla os novos resultados corrigidos de volta na planilha original.

### 🖥️ Dashboard Moderno
- **Interface Visual**: Frontend em React com Design System "Liquid".
- **Mapa Interativo**: Visualize os pontos encontrados Vs. pontos originais.
- **File Center**: Gestão centralizada dos arquivos gerados.

---

## 🚀 Instalação e Configuração

### Pré-requisitos
- Python 3.9+
- Node.js 18+
- Chave de API DeepSeek (Opcional, mas recomendado para alta precisão)

### 1. Configuração do Backend
```bash
# Clone o repositório
git clone https://github.com/gousero-sin/MapSearch.git
cd MapSearch

# (Opcional) Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows

# Instale as dependências
pip install -r requirements.txt
playwright install chromium
```

Crie um arquivo `.env` na raiz do projeto:
```ini
DEEPSEEK_API_KEY=sua_chave_aqui
```

### 2. Configuração do Frontend
```bash
cd frontend

# Instale as dependências
npm install

# Inicie o servidor de desenvolvimento
npm run dev
```

---

## ▶️ Como Usar

1.  **Inicie o Backend**:
    ```bash
    # Na raiz do projeto
    python api.py
    ```
2.  **Abra o Frontend**:
    Acesse `http://localhost:5173` (ou a porta indicada pelo Vite).
3.  **Processamento**:
    -   Clique em **"Buscar Maps"** para fazer upload de uma planilha Excel (`.xlsx`).
    -   A planilha deve ter uma coluna com os endereços (o sistema tenta detectar automaticamente colunas como `Completo`, `Endereço`, ou a 4ª coluna).
4.  **Acompanhamento**:
    -   Uma barra de progresso em tempo real mostrará o status.
    -   Resultados aparecem na tabela e no mapa instantaneamente.
5.  **Validação e Exportação**:
    -   Use o botão **"Validar Encontrados"** para rodar uma verificação profunda.
    -   Baixe o resultado final enriquecido com Latitude, Longitude, Score e Links do Maps.

---

## 🛠️ Tecnologias

-   **Backend**: Python, FastAPI, Playwright, Pandas, ThreadPoolExecutor.
-   **Frontend**: React, Vite, TailwindCSS, Framer Motion, Leaflet Maps.
-   **AI**: DeepSeek API (Semantic Matching & Parsing).

## 📄 Licença

**Uso Não Comercial (Gratuito) e Uso Comercial Restrito.**
Este software é gratuito para uso pessoal, educativo ou interno (não comercial). O uso para venda de serviços, integração em produtos comerciais ou distribuição paga é proibido sem autorização expressa.

Consulte o arquivo [LICENSE](./LICENSE) para mais detalhes.

---
Copyright © 2025 - Desenvolvido por [Seu Nome/Empresa].
