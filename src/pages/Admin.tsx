import { useMemo } from "react";
import { Users, TrendingUp, AlertTriangle, CheckCircle, XCircle, Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { mockCases, Case } from "@/data/mockData";
import { format } from "date-fns";
import { ptBR } from "date-fns/locale";

interface AuditorStats {
  nome: string;
  ocorrencias: number;
  emAnalise: number;
  naoFraude: number;
  fraude: number;
  valorTotal: number;
}

const mockAuditores = [
  "João Silva",
  "Maria Santos", 
  "Carlos Oliveira",
  "Ana Costa",
  "Pedro Lima"
];

export default function Admin() {
  const today = format(new Date(), "EEEE, d 'de' MMMM", { locale: ptBR });

  const auditorStats = useMemo((): AuditorStats[] => {
    const statsMap = new Map<string, AuditorStats>();

    mockAuditores.forEach(nome => {
      statsMap.set(nome, {
        nome,
        ocorrencias: 0,
        emAnalise: 0,
        naoFraude: 0,
        fraude: 0,
        valorTotal: 0
      });
    });

    mockCases.forEach((caso: Case) => {
      const auditor = caso.responsavel;
      if (statsMap.has(auditor)) {
        const stats = statsMap.get(auditor)!;
        
        switch (caso.status) {
          case 'aberto':
            stats.ocorrencias += 1;
            break;
          case 'em_analise':
            stats.emAnalise += 1;
            break;
          case 'pendente_resposta':
            stats.naoFraude += 1;
            break;
          case 'concluido':
          case 'arquivado':
            stats.fraude += 1;
            break;
        }
        
        stats.valorTotal += caso.valorTotalEnvolvido;
      }
    });

    return Array.from(statsMap.values());
  }, []);

  const totals = useMemo(() => {
    return auditorStats.reduce(
      (acc, auditor) => ({
        ocorrencias: acc.ocorrencias + auditor.ocorrencias,
        emAnalise: acc.emAnalise + auditor.emAnalise,
        naoFraude: acc.naoFraude + auditor.naoFraude,
        fraude: acc.fraude + auditor.fraude,
        valorTotal: acc.valorTotal + auditor.valorTotal
      }),
      { ocorrencias: 0, emAnalise: 0, naoFraude: 0, fraude: 0, valorTotal: 0 }
    );
  }, [auditorStats]);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-bold text-foreground">Painel Admin</h1>
        <p className="text-muted-foreground capitalize">{today}</p>
        <p className="text-sm text-muted-foreground mt-1">
          Visão estratégica das ocorrências por auditor
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10">
                <AlertTriangle className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Ocorrências</p>
                <p className="text-xl font-bold text-foreground">{totals.ocorrencias}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-amber-500/10">
                <Clock className="h-5 w-5 text-amber-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Em Análise</p>
                <p className="text-xl font-bold text-foreground">{totals.emAnalise}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-500/10">
                <CheckCircle className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Não Fraude</p>
                <p className="text-xl font-bold text-foreground">{totals.naoFraude}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-red-500/10">
                <XCircle className="h-5 w-5 text-red-500" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Fraude</p>
                <p className="text-xl font-bold text-foreground">{totals.fraude}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="bg-card border-border">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <TrendingUp className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Valor Total</p>
                <p className="text-lg font-bold text-foreground">R$ {totals.valorTotal.toFixed(2)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Auditors Table */}
      <Card className="bg-card border-border">
        <CardHeader>
          <CardTitle className="text-foreground flex items-center gap-2">
            <Users className="h-5 w-5" />
            Desempenho por Auditor
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground">Auditor</th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">
                    <div className="flex items-center justify-center gap-1">
                      <AlertTriangle className="h-4 w-4 text-blue-500" />
                      Ocorrência
                    </div>
                  </th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">
                    <div className="flex items-center justify-center gap-1">
                      <Clock className="h-4 w-4 text-amber-500" />
                      Em Análise
                    </div>
                  </th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">
                    <div className="flex items-center justify-center gap-1">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      Não Fraude
                    </div>
                  </th>
                  <th className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">
                    <div className="flex items-center justify-center gap-1">
                      <XCircle className="h-4 w-4 text-red-500" />
                      Fraude
                    </div>
                  </th>
                  <th className="text-right py-3 px-4 text-sm font-medium text-muted-foreground">Valor Total</th>
                </tr>
              </thead>
              <tbody>
                {auditorStats.map((auditor, index) => {
                  const total = auditor.ocorrencias + auditor.emAnalise + auditor.naoFraude + auditor.fraude;
                  return (
                    <tr 
                      key={auditor.nome} 
                      className={`border-b border-border/50 hover:bg-secondary/30 transition-colors ${
                        index % 2 === 0 ? 'bg-secondary/10' : ''
                      }`}
                    >
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                            <span className="text-xs font-medium text-primary">
                              {auditor.nome.split(' ').map(n => n[0]).join('')}
                            </span>
                          </div>
                          <span className="text-sm font-medium text-foreground">{auditor.nome}</span>
                        </div>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-flex items-center justify-center min-w-[28px] px-2 py-1 rounded-full text-sm font-medium ${
                          auditor.ocorrencias > 0 ? 'bg-blue-500/20 text-blue-400' : 'text-muted-foreground'
                        }`}>
                          {auditor.ocorrencias}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-flex items-center justify-center min-w-[28px] px-2 py-1 rounded-full text-sm font-medium ${
                          auditor.emAnalise > 0 ? 'bg-amber-500/20 text-amber-400' : 'text-muted-foreground'
                        }`}>
                          {auditor.emAnalise}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-flex items-center justify-center min-w-[28px] px-2 py-1 rounded-full text-sm font-medium ${
                          auditor.naoFraude > 0 ? 'bg-green-500/20 text-green-400' : 'text-muted-foreground'
                        }`}>
                          {auditor.naoFraude}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-center">
                        <span className={`inline-flex items-center justify-center min-w-[28px] px-2 py-1 rounded-full text-sm font-medium ${
                          auditor.fraude > 0 ? 'bg-red-500/20 text-red-400' : 'text-muted-foreground'
                        }`}>
                          {auditor.fraude}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right">
                        <span className="text-sm font-medium text-primary">
                          R$ {auditor.valorTotal.toFixed(2)}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              <tfoot>
                <tr className="bg-secondary/30 font-medium">
                  <td className="py-3 px-4 text-sm text-foreground">Total</td>
                  <td className="py-3 px-4 text-center text-sm text-blue-400">{totals.ocorrencias}</td>
                  <td className="py-3 px-4 text-center text-sm text-amber-400">{totals.emAnalise}</td>
                  <td className="py-3 px-4 text-center text-sm text-green-400">{totals.naoFraude}</td>
                  <td className="py-3 px-4 text-center text-sm text-red-400">{totals.fraude}</td>
                  <td className="py-3 px-4 text-right text-sm text-primary">R$ {totals.valorTotal.toFixed(2)}</td>
                </tr>
              </tfoot>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
