import { useState } from "react";
import { FileText, Download, FileSpreadsheet, FileJson, Calendar, Store, Filter } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { mockStores, mockAlerts } from "@/data/mockData";
import { AlertBadge } from "@/components/shared/AlertBadge";
import { StatusBadge } from "@/components/shared/StatusBadge";
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
import { toast } from "sonner";

const reportTypes = [
  {
    id: "alertas",
    title: "Relatório de Alertas",
    description: "Exportar todos os alertas do período selecionado",
    icon: FileText,
    color: "text-blue-500"
  },
  {
    id: "casos",
    title: "Relatório de Casos",
    description: "Exportar investigações e seu status",
    icon: FileText,
    color: "text-purple-500"
  },
  {
    id: "trocas",
    title: "Relatório de Trocas",
    description: "Exportar análise de trocas suspeitas",
    icon: FileText,
    color: "text-amber-500"
  },
  {
    id: "ajustes",
    title: "Relatório de Ajustes",
    description: "Exportar movimentações de estoque",
    icon: FileText,
    color: "text-emerald-500"
  }
];

export default function Relatorios() {
  const [selectedReport, setSelectedReport] = useState<string>("alertas");
  const [dateStart, setDateStart] = useState("");
  const [dateEnd, setDateEnd] = useState("");
  const [selectedStore, setSelectedStore] = useState("all");

  const handleExport = (format: string) => {
    toast.success(`Exportando relatório em formato ${format.toUpperCase()}...`);
  };

  const formatDate = (date: Date) => {
    return format(date, "dd/MM/yyyy HH:mm", { locale: ptBR });
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Relatórios</h1>
        <p className="text-muted-foreground">Gere e exporte relatórios de análise</p>
      </div>

      {/* Report Type Selection */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {reportTypes.map((report) => (
          <Card 
            key={report.id}
            className={`bg-card border-border cursor-pointer transition-all hover:border-primary/50 ${
              selectedReport === report.id ? 'border-primary ring-1 ring-primary' : ''
            }`}
            onClick={() => setSelectedReport(report.id)}
          >
            <CardContent className="pt-4">
              <report.icon className={`h-8 w-8 ${report.color} mb-3`} />
              <h3 className="font-medium text-foreground">{report.title}</h3>
              <p className="text-xs text-muted-foreground mt-1">{report.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Filters */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Filtros
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label className="text-muted-foreground">Data Início</Label>
              <Input
                type="date"
                value={dateStart}
                onChange={(e) => setDateStart(e.target.value)}
                className="bg-background border-border"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-muted-foreground">Data Fim</Label>
              <Input
                type="date"
                value={dateEnd}
                onChange={(e) => setDateEnd(e.target.value)}
                className="bg-background border-border"
              />
            </div>
            <div className="space-y-2">
              <Label className="text-muted-foreground">Loja</Label>
              <Select value={selectedStore} onValueChange={setSelectedStore}>
                <SelectTrigger className="bg-background border-border">
                  <SelectValue placeholder="Selecione a loja" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Todas as Lojas</SelectItem>
                  {mockStores.map(store => (
                    <SelectItem key={store.id} value={store.id}>{store.nome}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label className="text-muted-foreground">Nível de Risco</Label>
              <Select defaultValue="all">
                <SelectTrigger className="bg-background border-border">
                  <SelectValue placeholder="Selecione" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Todos</SelectItem>
                  <SelectItem value="critico">Crítico</SelectItem>
                  <SelectItem value="medio">Médio</SelectItem>
                  <SelectItem value="leve">Leve</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Preview */}
      <Card className="bg-card border-border">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-lg">Preview do Relatório</CardTitle>
            <CardDescription>Visualização prévia dos dados que serão exportados</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => handleExport('pdf')}>
              <Download className="h-4 w-4 mr-2" />
              PDF
            </Button>
            <Button variant="outline" size="sm" onClick={() => handleExport('excel')}>
              <FileSpreadsheet className="h-4 w-4 mr-2" />
              Excel
            </Button>
            <Button variant="outline" size="sm" onClick={() => handleExport('csv')}>
              <FileJson className="h-4 w-4 mr-2" />
              CSV
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="border-border hover:bg-transparent">
                <TableHead className="text-muted-foreground">Data</TableHead>
                <TableHead className="text-muted-foreground">Loja</TableHead>
                <TableHead className="text-muted-foreground">Tipo</TableHead>
                <TableHead className="text-muted-foreground">Descrição</TableHead>
                <TableHead className="text-muted-foreground">Risco</TableHead>
                <TableHead className="text-muted-foreground">Status</TableHead>
                <TableHead className="text-muted-foreground text-right">Valor</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {mockAlerts.slice(0, 5).map((alert) => (
                <TableRow key={alert.id} className="border-border">
                  <TableCell className="text-sm text-foreground">
                    {formatDate(alert.createdAt)}
                  </TableCell>
                  <TableCell className="text-sm text-foreground">
                    {alert.lojaNome}
                  </TableCell>
                  <TableCell className="text-sm text-foreground capitalize">
                    {alert.tipo}
                  </TableCell>
                  <TableCell className="text-sm text-foreground max-w-[200px] truncate">
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
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <div className="mt-4 text-center text-sm text-muted-foreground">
            Mostrando 5 de {mockAlerts.length} registros
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
