import { AlertTriangle, FolderOpen, TrendingUp, CheckCircle, Store, Clock } from "lucide-react";
import { StatCard } from "@/components/dashboard/StatCard";
import { AlertBadge } from "@/components/shared/AlertBadge";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { 
  mockKPIs, 
  alertsByStoreData, 
  riskDistributionData, 
  alertsTrendData,
  mockAlerts,
  mockCases 
} from "@/data/mockData";
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
  Line
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { format } from "date-fns";
import { ptBR } from "date-fns/locale";

export default function Dashboard() {
  const today = format(new Date(), "EEEE, d 'de' MMMM", { locale: ptBR });
  const recentAlerts = mockAlerts.filter(a => a.status === 'pendente').slice(0, 5);
  const recentCases = mockCases.filter(c => c.status !== 'concluido' && c.status !== 'arquivado').slice(0, 4);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
        <p className="text-muted-foreground capitalize">{today}</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Total de Alertas"
          value={mockKPIs.totalAlertasHoje}
          subtitle="Hoje"
          icon={AlertTriangle}
          trend={{ value: 12, isPositive: false }}
        />
        <StatCard
          title="Alertas Críticos"
          value={mockKPIs.alertasCriticos}
          subtitle="Requerem atenção imediata"
          icon={AlertTriangle}
          variant="danger"
        />
        <StatCard
          title="Casos em Aberto"
          value={mockKPIs.casosAbertos}
          subtitle="Investigações ativas"
          icon={FolderOpen}
          variant="warning"
        />
        <StatCard
          title="Taxa de Resolução"
          value={`${mockKPIs.taxaResolucao}%`}
          subtitle="Últimos 30 dias"
          icon={CheckCircle}
          variant="success"
          trend={{ value: 5, isPositive: true }}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Alerts by Store */}
        <Card className="lg:col-span-2 bg-card border-border">
          <CardHeader>
            <CardTitle className="text-foreground flex items-center gap-2">
              <Store className="h-5 w-5" />
              Alertas por Loja (últimos 7 dias)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={alertsByStoreData}>
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
                <Bar dataKey="alertas" fill="#3b82f6" name="Total" radius={[4, 4, 0, 0]} />
                <Bar dataKey="criticos" fill="#ef4444" name="Críticos" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Risk Distribution */}
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
                  <span className="text-xs text-muted-foreground">{item.name}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Trend Chart */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Tendência de Alertas (últimos 30 dias)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={alertsTrendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="data" stroke="#888" fontSize={11} />
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
                dataKey="alertas" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={{ fill: '#3b82f6', strokeWidth: 0, r: 3 }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Recent Alerts and Cases */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Recent Critical Alerts */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="text-foreground flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-500" />
              Alertas Recentes Pendentes
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {recentAlerts.map((alert) => (
              <div 
                key={alert.id} 
                className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{alert.titulo}</p>
                  <p className="text-xs text-muted-foreground">{alert.lojaNome}</p>
                </div>
                <div className="flex items-center gap-2">
                  <AlertBadge nivel={alert.nivelRisco} />
                  <span className="text-xs text-muted-foreground">
                    R$ {alert.valorEnvolvido.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Recent Cases */}
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="text-foreground flex items-center gap-2">
              <Clock className="h-5 w-5 text-amber-500" />
              Casos em Andamento
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {recentCases.map((caso) => (
              <div 
                key={caso.id} 
                className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">{caso.titulo}</p>
                  <p className="text-xs text-muted-foreground">
                    {caso.lojaNome} • {caso.alertasVinculados} alertas
                  </p>
                </div>
                <StatusBadge status={caso.status} />
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
