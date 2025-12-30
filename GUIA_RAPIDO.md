# 🚀 Guia Rápido - Sentinela Data Mart

## ✅ Seu sistema JÁ consulta direto no PostgreSQL!

A arquitetura está implementada conforme o diagrama:
- ✓ FastAPI conecta direto no PostgreSQL
- ✓ Tools IA consultam as tabelas do Data Mart
- ✓ Sem CSV intermediário
- ✓ Queries otimizadas

---

## 📋 Passo a Passo

### 1️⃣ Teste a conexão

```bash
python testar_conexao.py
```

Deve mostrar:
- ✓ Conexão estabelecida
- ✓ Lista de tabelas encontradas
- ✓ Exemplo de consulta

---

### 2️⃣ Configure sua chave OpenAI

Edite o `.env` e adicione:

```env
OPENAI_API_KEY=sk-proj-...sua-chave-aqui...
```

---

### 3️⃣ Inicie o servidor

```bash
python server.py --host 127.0.0.1 --port 8000
```

Ou use o atalho:
```bash
iniciar.bat
```

---

### 4️⃣ Acesse a interface

Abra no navegador:
- **UI Chat:** http://127.0.0.1:8000/
- **Swagger:** http://127.0.0.1:8000/docs

---

## 💬 Exemplos de Prompts

### Listar tabelas
```
Liste as tabelas disponíveis
```

### Preview de dados
```
Mostre ajuste_estoque_2025 com 10 linhas
```

### Filtros
```
Filtre loja=17 em ajuste_estoque_2025 limite 20
```

### Agregações
```
Agregue em cancelamento_2025 por loja somando valor top 10
```

### Contagens
```
Conte quantas vezes sku=12345 aparece em troca_2025
```

### Análise de período
```
Mostre vendas de janeiro a março 2025 da loja 17
```

### Cruzamento de dados
```
Quais SKUs aparecem tanto em ajustes quanto em devoluções da loja 17?
```

---

## 🎯 Tabelas Esperadas (conforme arquitetura)

- `ajuste_estoque_2025` - Ajustes de estoque
- `cancelamento_2025` - Cancelamentos
- `inventario_saida_2025` - Saídas de inventário
- `troca_2025` - Trocas/devoluções

O sistema detecta automaticamente qualquer tabela no schema `public`.

---

## 🔧 Estrutura do Banco

### Colunas típicas detectadas automaticamente:

- **Loja:** `loja`, `filial`, `cod_loja`
- **SKU:** `sku`, `produto`, `codigo_produto`
- **Data:** `data`, `datacancelamento`, `dt_mov`
- **Valor:** `valor`, `valorbruto`, `preco`
- **Quantidade:** `qtd`, `quantidade`, `qtde`

---

## 📊 Endpoints da API

### Status
- `GET /status` - Status do servidor
- `GET /ping` - Health check
- `GET /stats` - Métricas em tempo real

### Dados
- `GET /tool/list_tables` - Lista tabelas
- `GET /tool/sql_head` - Preview de tabela
- `GET /tool/sql_filter` - Filtros complexos
- `GET /tool/sql_aggregate` - Agregações
- `GET /tool/sql_count` - Contagens

---

## 🐛 Troubleshooting

### Erro: "Falha na conexão com PostgreSQL"
→ Verifique se o PostgreSQL está rodando
→ Confirme credenciais no `.env`

### Erro: "OPENAI_API_KEY ausente"
→ Configure a chave no `.env`

### Erro: "Nenhuma tabela encontrada"
→ Verifique se as tabelas existem no schema `public`
→ Execute: `SELECT * FROM information_schema.tables WHERE table_schema='public'`

---

## 📁 Arquivos Importantes

- `.env` - Configurações (banco + IA)
- `agent_app.py` - Aplicação FastAPI
- `server.py` - Launcher
- `testar_conexao.py` - Script de teste
- `iniciar.bat` - Atalho Windows

---

## 🎓 Como Funciona

1. **Usuário** envia prompt em linguagem natural
2. **Planner IA** analisa e decide qual tool usar
3. **Tool** executa query SQL no PostgreSQL
4. **Resultado** é formatado e retornado ao usuário

Tudo em tempo real, sem arquivos intermediários! 🚀
