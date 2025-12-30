import { TrendingUp, Store, Clock, AlertTriangle } from "lucide-react";
import { StatCard } from "@/components/dashboard/StatCard";
import { PriorityBadge } from "@/components/shared/PriorityBadge";
import { mockCases, Case } from "@/data/mockData";
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  LabelList
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { format } from "date-fns";
import { ptBR } from "date-fns/locale";
import { useMemo } from "react";

export default function Dashboard() {
  const today = format(new Date(), "EEEE, d 'de' MMMM", { locale: ptBR });

  const riskStats = useMemo(() => {
    const baixo = mockCases.filter(c => c.prioridade === 'baixo');
    const medio = mockCases.filter(c => c.prioridade === 'médio');
    const critico = mockCases.filter(c => c.prioridade === 'crítico');

    return {
      baixo: {
        count: baixo.length,
        total: baixo.reduce((acc, c) => acc + c.valorTotalEnvolvido, 0)
      },
      medio: {
        count: medio.length,
        total: medio.reduce((acc, c) => acc + c.valorTotalEnvolvido, 0)
      },
      critico: {
        count: critico.length,
        total: critico.reduce((acc, c) => acc + c.valorTotalEnvolvido, 0)
      },
      totalGeral: mockCases.reduce((acc, c) => acc + c.valorTotalEnvolvido, 0),
      totalCount: mockCases.length
    };
  }, []);

  const ocorrenciasPorLoja = useMemo(() => {
    const lojaMap = new Map<string, { total: number; criticos: number }>();
    
    mockCases.forEach(c => {
      const current = lojaMap.get(c.lojaNome) || { total: 0, criticos: 0 };
      current.total += 1;
      if (c.prioridade === 'crítico') current.criticos += 1;
      lojaMap.set(c.lojaNome, current);
    });

    return Array.from(lojaMap.entries()).map(([loja, data]) => ({
      loja: loja.replace('Loja ', ''),
      ocorrencias: data.total,
      criticos: data.criticos
    }));
  }, []);

  const riskDistributionData = useMemo(() => [
    { name: 'Crítico', value: riskStats.critico.count, total: riskStats.critico.total, color: '#ef4444' },
    { name: 'Médio', value: riskStats.medio.count, total: riskStats.medio.total, color: '#f59e0b' },
    { name: 'Baixo', value: riskStats.baixo.count, total: riskStats.baixo.total, color: '#3b82f6' }
  ], [riskStats]);

  const tendenciaOcorrencias = useMemo(() => {
    const last30Days: { [key: string]: { baixo: number; medio: number; critico: number } } = {};
    
    for (let i = 29; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      const dateKey = format(date, 'dd/MM');
      last30Days[dateKey] = { baixo: 0, medio: 0, critico: 0 };
    }

    mockCases.forEach(c => {
      const dateKey = format(c.createdAt, 'dd/MM');
      if (last30Days[dateKey]) {
        if (c.prioridade === 'baixo') last30Days[dateKey].baixo += 1;
        else if (c.prioridade === 'médio') last30Days[dateKey].medio += 1;
        else if (c.prioridade === 'crítico') last30Days[dateKey].critico += 1;
      }
    });

    return Object.entries(last30Days).map(([data, values]) => ({
      data,
      ...values,
      total: values.baixo + values.medio + values.critico
    }));
  }, []);

  const recentOcorrencias = useMemo(() => {
    return [...mockCases]
      .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
      .slice(0, 5);
  }, []);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
        <p className="text-muted-foreground capitalize">{today}</p>
      </div>

      {/* KPI Cards - Total por Risco */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Risco Baixo"
          value={riskStats.baixo.count}
          subtitle={`R$ ${riskStats.baixo.total.toFixed(2)}`}
          icon={AlertTriangle}
          variant="default"
        />
        <StatCard
          title="Risco Médio"
          value={riskStats.medio.count}
          subtitle={`R$ ${riskStats.medio.total.toFixed(2)}`}
          icon={AlertTriangle}
          variant="warning"
        />
        <StatCard
          title="Risco Crítico"
          value={riskStats.critico.count}
          subtitle={`R$ ${riskStats.critico.total.toFixed(2)}`}
          icon={AlertTriangle}
          variant="danger"
        />
        <StatCard
          title="Total Geral"
          value={riskStats.totalCount}
          subtitle={`R$ ${riskStats.totalGeral.toFixed(2)}`}
          icon={TrendingUp}
          variant="success"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Ocorrências por Loja */}
        <Card className="lg:col-span-2 bg-card border-border">
          <CardHeader>
            <CardTitle className="text-foreground flex items-center gap-2">
              <Store className="h-5 w-5" />
              Ocorrências por Loja
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={ocorrenciasPorLoja}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis dataKey="loja" stroke="#888" fontSize={12} />
                <YAxis stroke="#888" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e2634', 
                    border: '1px solid #333',
                    borderRadius: '8px'
                  }}
                />
                <Bar dataKey="ocorrencias" fill="#3b82f6" name="Total" radius={[4, 4, 0, 0]} />
                <Bar dataKey="criticos" fill="#ef4444" name="Críticos" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Risk Distribution com valores */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="text-foreground">Distribuição por Risco</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={riskDistributionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, value }) => `${value}`}
                  labelLine={false}
                >
                  {riskDistributionData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1e2634', 
                    border: '1px solid #333',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number, name: string, props: any) => [
                    `${value} ocorrências (R$ ${props.payload.total.toFixed(2)})`,
                    name
                  ]}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex justify-center gap-4 mt-2">
              {riskDistributionData.map((item) => (
                <div key={item.name} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: item.color }} 
                  />
                  <span className="text-xs text-slate-200">
                    {item.name}: {item.value} (R$ {item.total.toFixed(2)})
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Trend Chart - Tendência de Ocorrências */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Tendência de Ocorrências (últimos 30 dias)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={tendenciaOcorrencias}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="data" stroke="#888" fontSize={11} interval={4} />
              <YAxis stroke="#888" fontSize={12} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e2634', 
                  border: '1px solid #333',
                  borderRadius: '8px'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="critico" 
                stroke="#ef4444" 
                strokeWidth={2}
                name="Crítico"
                dot={{ fill: '#ef4444', strokeWidth: 0, r: 2 }}
              />
              <Line 
                type="monotone" 
                dataKey="medio" 
                stroke="#f59e0b" 
                strokeWidth={2}
                name="Médio"
                dot={{ fill: '#f59e0b', strokeWidth: 0, r: 2 }}
              />
              <Line 
                type="monotone" 
                dataKey="baixo" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Baixo"
                dot={{ fill: '#3b82f6', strokeWidth: 0, r: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Ocorrências Recentes */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <Clock className="h-5 w-5 text-amber-500" />
            Ocorrências Recentes
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {recentOcorrencias.map((caso) => (
            <div 
              key={caso.id} 
              className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg"
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">{caso.titulo}</p>
                <p className="text-xs text-muted-foreground">
                  {caso.lojaNome} • {format(caso.createdAt, "dd/MM/yyyy HH:mm", { locale: ptBR })}
                </p>
              </div>
              <div className="flex items-center gap-3">
                <PriorityBadge prioridade={caso.prioridade} />
                <span className="text-sm font-medium text-primary">
                  R$ {caso.valorTotalEnvolvido.toFixed(2)}
                </span>
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
