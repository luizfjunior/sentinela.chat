import { useState } from "react";
import { AlertTriangle, Filter, Search, Eye, FolderPlus, CheckCircle } from "lucide-react";
import { mockAlerts, mockStores, Alert } from "@/data/mockData";
import { AlertBadge } from "@/components/shared/AlertBadge";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { format } from "date-fns";
import { ptBR } from "date-fns/locale";

export default function Alertas() {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedStore, setSelectedStore] = useState<string>("all");
  const [selectedStatus, setSelectedStatus] = useState<string>("all");
  const [activeTab, setActiveTab] = useState("todos");

  const filteredAlerts = mockAlerts.filter(alert => {
    const matchesSearch = alert.titulo.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         alert.descricao.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         alert.lojaNome.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStore = selectedStore === "all" || alert.lojaId === selectedStore;
    const matchesStatus = selectedStatus === "all" || alert.status === selectedStatus;
    const matchesTab = activeTab === "todos" || 
                      (activeTab === "criticos" && alert.nivelRisco === "critico") ||
                      (activeTab === "medios" && alert.nivelRisco === "medio") ||
                      (activeTab === "leves" && alert.nivelRisco === "leve");
    
    return matchesSearch && matchesStore && matchesStatus && matchesTab;
  });

  const countByRisk = {
    todos: mockAlerts.length,
    criticos: mockAlerts.filter(a => a.nivelRisco === 'critico').length,
    medios: mockAlerts.filter(a => a.nivelRisco === 'medio').length,
    leves: mockAlerts.filter(a => a.nivelRisco === 'leve').length
  };

  const formatDate = (date: Date) => {
    return format(date, "dd/MM/yyyy HH:mm", { locale: ptBR });
  };

  const getTipoLabel = (tipo: string) => {
    const labels: Record<string, string> = {
      troca: 'Troca',
      ajuste: 'Ajuste',
      cancelamento: 'Cancelamento',
      cruzamento: 'Cruzamento'
    };
    return labels[tipo] || tipo;
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Alertas</h1>
          <p className="text-muted-foreground">Gerencie e analise alertas de fraude</p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-foreground">{countByRisk.todos}</div>
            <p className="text-xs text-muted-foreground">Total de Alertas</p>
          </CardContent>
        </Card>
        <Card className="bg-red-500/10 border-red-500/20">
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-red-500">{countByRisk.criticos}</div>
            <p className="text-xs text-red-400">Críticos</p>
          </CardContent>
        </Card>
        <Card className="bg-amber-500/10 border-amber-500/20">
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-amber-500">{countByRisk.medios}</div>
            <p className="text-xs text-amber-400">Médios</p>
          </CardContent>
        </Card>
        <Card className="bg-emerald-500/10 border-emerald-500/20">
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-emerald-500">{countByRisk.leves}</div>
            <p className="text-xs text-emerald-400">Leves</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters */}
      <Card className="bg-card border-border">
        <CardContent className="pt-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Buscar alertas..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9 bg-background border-border"
              />
            </div>
            <Select value={selectedStore} onValueChange={setSelectedStore}>
              <SelectTrigger className="w-full sm:w-[180px] bg-background border-border">
                <SelectValue placeholder="Loja" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Todas as Lojas</SelectItem>
                {mockStores.map(store => (
                  <SelectItem key={store.id} value={store.id}>{store.nome}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={selectedStatus} onValueChange={setSelectedStatus}>
              <SelectTrigger className="w-full sm:w-[180px] bg-background border-border">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Todos os Status</SelectItem>
                <SelectItem value="pendente">Pendente</SelectItem>
                <SelectItem value="em_analise">Em Análise</SelectItem>
                <SelectItem value="resolvido">Resolvido</SelectItem>
                <SelectItem value="falso_positivo">Falso Positivo</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Tabs and Table */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="bg-secondary">
          <TabsTrigger value="todos">Todos ({countByRisk.todos})</TabsTrigger>
          <TabsTrigger value="criticos" className="text-red-500">Críticos ({countByRisk.criticos})</TabsTrigger>
          <TabsTrigger value="medios" className="text-amber-500">Médios ({countByRisk.medios})</TabsTrigger>
          <TabsTrigger value="leves" className="text-emerald-500">Leves ({countByRisk.leves})</TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="mt-4">
          <Card className="bg-card border-border">
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow className="border-border hover:bg-transparent">
                    <TableHead className="text-muted-foreground">Data/Hora</TableHead>
                    <TableHead className="text-muted-foreground">Loja</TableHead>
                    <TableHead className="text-muted-foreground">Tipo</TableHead>
                    <TableHead className="text-muted-foreground">Descrição</TableHead>
                    <TableHead className="text-muted-foreground">Risco</TableHead>
                    <TableHead className="text-muted-foreground">Status</TableHead>
                    <TableHead className="text-muted-foreground text-right">Valor</TableHead>
                    <TableHead className="text-muted-foreground text-right">Ações</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredAlerts.map((alert) => (
                    <TableRow key={alert.id} className="border-border">
                      <TableCell className="text-sm text-foreground">
                        {formatDate(alert.createdAt)}
                      </TableCell>
                      <TableCell className="text-sm text-foreground">
                        {alert.lojaNome}
                      </TableCell>
                      <TableCell className="text-sm text-foreground">
                        {getTipoLabel(alert.tipo)}
                      </TableCell>
                      <TableCell className="text-sm text-foreground max-w-[250px] truncate">
                        {alert.titulo}
                      </TableCell>
                      <TableCell>
                        <AlertBadge nivel={alert.nivelRisco} />
                      </TableCell>
                      <TableCell>
                        <StatusBadge status={alert.status} />
                      </TableCell>
                      <TableCell className="text-sm text-foreground text-right">
                        R$ {alert.valorEnvolvido.toFixed(2)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1">
                          <Button variant="ghost" size="icon" className="h-8 w-8" title="Ver detalhes">
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-8 w-8" title="Criar caso">
                            <FolderPlus className="h-4 w-4" />
                          </Button>
                          <Button variant="ghost" size="icon" className="h-8 w-8" title="Resolver">
                            <CheckCircle className="h-4 w-4" />
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
              {filteredAlerts.length === 0 && (
                <div className="py-12 text-center text-muted-foreground">
                  Nenhum alerta encontrado com os filtros selecionados
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
