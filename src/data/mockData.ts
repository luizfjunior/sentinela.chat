// Mock data for Sentinela 2.0

export interface Alert {
  id: string;
  lojaId: string;
  lojaNome: string;
  tipo: 'troca' | 'ajuste' | 'cancelamento' | 'cruzamento';
  nivelRisco: 'baixo' | 'médio' | 'crítico';
  status: 'pendente' | 'em_analise' | 'resolvido' | 'falso_positivo';
  titulo: string;
  descricao: string;
  produtoSku?: string;
  produtoNome?: string;
  valorEnvolvido: number;
  createdAt: Date;
  resolvedAt?: Date;
}

export interface Case {
  id: string;
  titulo: string;
  descricao: string;
  status: 'aberto' | 'em_analise' | 'pendente_resposta' | 'concluido' | 'arquivado';
  prioridade: 'baixo' | 'médio' | 'crítico';
  lojaId: string;
  lojaNome: string;
  alertasVinculados: number;
  valorTotalEnvolvido: number;
  responsavel: string;
  createdAt: Date;
  closedAt?: Date;
}

export interface Store {
  id: string;
  nome: string;
  alertasHoje: number;
  alertasSemana: number;
}

// Mock Alerts
export const mockAlerts: Alert[] = [
  {
    id: '1',
    lojaId: 'loja-001',
    lojaNome: 'Loja Centro',
    tipo: 'troca',
    nivelRisco: 'crítico',
    status: 'pendente',
    titulo: 'Troca suspeita - CPF diferente do comprador',
    descricao: 'Cliente realizando troca de produto comprado por terceiro há mais de 90 dias',
    produtoSku: '999999992529538',
    produtoNome: 'Tênis Nike Air Max',
    valorEnvolvido: 599.90,
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000)
  },
  {
    id: '2',
    lojaId: 'loja-002',
    lojaNome: 'Loja Shopping',
    tipo: 'ajuste',
    nivelRisco: 'crítico',
    status: 'em_analise',
    titulo: 'Ajuste entrada/saída sequencial',
    descricao: 'Produto teve ajuste de entrada em 27/03 e saída em 28/03',
    produtoSku: '999999992529538',
    produtoNome: 'Bolsa Feminina',
    valorEnvolvido: 289.90,
    createdAt: new Date(Date.now() - 5 * 60 * 60 * 1000)
  },
  {
    id: '3',
    lojaId: 'loja-003',
    lojaNome: 'Loja Norte',
    tipo: 'troca',
    nivelRisco: 'médio',
    status: 'pendente',
    titulo: 'Troca entre departamentos diferentes',
    descricao: 'Troca de meias por calçado com diferença negativa',
    produtoSku: '888777665544',
    produtoNome: 'Meia Esportiva',
    valorEnvolvido: 89.90,
    createdAt: new Date(Date.now() - 8 * 60 * 60 * 1000)
  },
  {
    id: '4',
    lojaId: 'loja-001',
    lojaNome: 'Loja Centro',
    tipo: 'cancelamento',
    nivelRisco: 'crítico',
    status: 'pendente',
    titulo: 'Cancelamento + inventário de saída',
    descricao: 'Produto cancelado teve saída de inventário 3 meses depois',
    produtoSku: '777666555444',
    produtoNome: 'Jaqueta Masculina',
    valorEnvolvido: 450.00,
    createdAt: new Date(Date.now() - 12 * 60 * 60 * 1000)
  },
  {
    id: '5',
    lojaId: 'loja-004',
    lojaNome: 'Loja Sul',
    tipo: 'cruzamento',
    nivelRisco: 'crítico',
    status: 'pendente',
    titulo: 'Cruzamento entre lojas',
    descricao: 'Produto faltou na Loja Norte e entrou como devolução na Loja Sul',
    produtoSku: '666555444333',
    produtoNome: 'Relógio Digital',
    valorEnvolvido: 320.00,
    createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000)
  },
  {
    id: '6',
    lojaId: 'loja-002',
    lojaNome: 'Loja Shopping',
    tipo: 'troca',
    nivelRisco: 'baixo',
    status: 'resolvido',
    titulo: 'Troca mesma marca com diferença',
    descricao: 'Troca normal dentro das regras',
    produtoSku: '555444333222',
    produtoNome: 'Camiseta Básica',
    valorEnvolvido: 25.00,
    createdAt: new Date(Date.now() - 48 * 60 * 60 * 1000),
    resolvedAt: new Date(Date.now() - 24 * 60 * 60 * 1000)
  },
  {
    id: '7',
    lojaId: 'loja-003',
    lojaNome: 'Loja Norte',
    tipo: 'ajuste',
    nivelRisco: 'médio',
    status: 'pendente',
    titulo: 'Múltiplos ajustes no mesmo produto',
    descricao: '5 ajustes de saída no mesmo SKU em 7 dias',
    produtoSku: '444333222111',
    produtoNome: 'Perfume Importado',
    valorEnvolvido: 890.00,
    createdAt: new Date(Date.now() - 72 * 60 * 60 * 1000)
  },
  {
    id: '8',
    lojaId: 'loja-001',
    lojaNome: 'Loja Centro',
    tipo: 'troca',
    nivelRisco: 'baixo',
    status: 'resolvido',
    titulo: 'Troca padrão aprovada',
    descricao: 'Troca dentro do prazo e condições normais',
    produtoSku: '333222111000',
    produtoNome: 'Calça Jeans',
    valorEnvolvido: 15.00,
    createdAt: new Date(Date.now() - 96 * 60 * 60 * 1000),
    resolvedAt: new Date(Date.now() - 80 * 60 * 60 * 1000)
  }
];

// Mock Cases
export const mockCases: Case[] = [
  {
    id: '1',
    titulo: 'Investigação Loja Centro - Padrão de Fraude',
    descricao: 'Múltiplos alertas críticos identificados na Loja Centro nas últimas 48h',
    status: 'em_analise',
    prioridade: 'crítico',
    lojaId: 'loja-001',
    lojaNome: 'Loja Centro',
    alertasVinculados: 3,
    valorTotalEnvolvido: 1049.90,
    responsavel: 'João Silva',
    createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000)
  },
  {
    id: '2',
    titulo: 'Cruzamento Loja Norte x Sul',
    descricao: 'Análise de produtos que faltaram em uma loja e apareceram em outra',
    status: 'aberto',
    prioridade: 'crítico',
    lojaId: 'loja-003',
    lojaNome: 'Loja Norte',
    alertasVinculados: 2,
    valorTotalEnvolvido: 1210.00,
    responsavel: 'Maria Santos',
    createdAt: new Date(Date.now() - 48 * 60 * 60 * 1000)
  },
  {
    id: '3',
    titulo: 'Ajustes suspeitos Shopping',
    descricao: 'Padrão de ajustes entrada/saída sequenciais identificado',
    status: 'pendente_resposta',
    prioridade: 'médio',
    lojaId: 'loja-002',
    lojaNome: 'Loja Shopping',
    alertasVinculados: 1,
    valorTotalEnvolvido: 289.90,
    responsavel: 'Carlos Oliveira',
    createdAt: new Date(Date.now() - 72 * 60 * 60 * 1000)
  },
  {
    id: '4',
    titulo: 'Cancelamentos antigos com saída',
    descricao: 'Produtos cancelados tiveram saída de inventário posteriormente',
    status: 'concluido',
    prioridade: 'baixo',
    lojaId: 'loja-001',
    lojaNome: 'Loja Centro',
    alertasVinculados: 1,
    valorTotalEnvolvido: 450.00,
    responsavel: 'Ana Costa',
    createdAt: new Date(Date.now() - 120 * 60 * 60 * 1000),
    closedAt: new Date(Date.now() - 24 * 60 * 60 * 1000)
  }
];

// Mock Stores
export const mockStores: Store[] = [
  { id: 'loja-001', nome: 'Loja Centro', alertasHoje: 3, alertasSemana: 12 },
  { id: 'loja-002', nome: 'Loja Shopping', alertasHoje: 2, alertasSemana: 8 },
  { id: 'loja-003', nome: 'Loja Norte', alertasHoje: 2, alertasSemana: 6 },
  { id: 'loja-004', nome: 'Loja Sul', alertasHoje: 1, alertasSemana: 5 },
  { id: 'loja-005', nome: 'Loja Oeste', alertasHoje: 0, alertasSemana: 3 }
];

// KPI Data
export const mockKPIs = {
  totalAlertasHoje: 8,
  alertasCriticos: 4,
  casosAbertos: 3,
  taxaResolucao: 72
};

// Chart Data
export const alertsByStoreData = [
  { loja: 'Centro', alertas: 12, criticos: 5 },
  { loja: 'Shopping', alertas: 8, criticos: 2 },
  { loja: 'Norte', alertas: 6, criticos: 1 },
  { loja: 'Sul', alertas: 5, criticos: 2 },
  { loja: 'Oeste', alertas: 3, criticos: 0 }
];

export const riskDistributionData = [
  { name: 'Crítico', value: 4, color: '#ef4444' },
  { name: 'Médio', value: 2, color: '#f59e0b' },
  { name: 'Baixo', value: 2, color: '#3b82f6' }
];

export const alertsTrendData = [
  { data: '01/12', alertas: 5 },
  { data: '02/12', alertas: 8 },
  { data: '03/12', alertas: 6 },
  { data: '04/12', alertas: 12 },
  { data: '05/12', alertas: 9 },
  { data: '06/12', alertas: 15 },
  { data: '07/12', alertas: 11 },
  { data: '08/12', alertas: 8 },
  { data: '09/12', alertas: 14 },
  { data: '10/12', alertas: 10 },
  { data: '11/12', alertas: 7 },
  { data: '12/12', alertas: 13 },
  { data: '13/12', alertas: 9 },
  { data: '14/12', alertas: 11 },
  { data: '15/12', alertas: 8 },
  { data: '16/12', alertas: 6 },
  { data: '17/12', alertas: 10 },
  { data: '18/12', alertas: 8 }
];
