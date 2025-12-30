import { useState } from "react";
import { Search, Play, Calendar, Store } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface AnalysisType {
  id: string;
  title: string;
  description: string;
  color: string;
  bgColor: string;
  borderColor: string;
}

const analysisTypes: AnalysisType[] = [
  {
    id: "leve",
    title: "Análise Leve",
    description: "Trocas da mesma marca e departamento, com diferença para pagar. Entrada: 1 item | Saída: 1 item",
    color: "text-emerald-500",
    bgColor: "bg-emerald-500/10",
    borderColor: "border-emerald-500/30"
  },
  {
    id: "medio",
    title: "Análise Média",
    description: "Trocas entre marcas/departamentos diferentes, sem diferença ou diferença negativa. Entrada: 1 item | Saída: até 3 itens",
    color: "text-amber-500",
    bgColor: "bg-amber-500/10",
    borderColor: "border-amber-500/30"
  },
  {
    id: "critico",
    title: "Análise Crítica",
    description: "Situações de alto risco: diferença ≤R$10, comprador ≠ pessoa da troca, produto >90 dias, meias por calçados",
    color: "text-red-500",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/30"
  }
];

interface AnalysisFilters {
  store: string;
  startDate: string;
  endDate: string;
}

export default function Alertas() {
  const [filters, setFilters] = useState<Record<string, AnalysisFilters>>({
    leve: { store: "", startDate: "", endDate: "" },
    medio: { store: "", startDate: "", endDate: "" },
    critico: { store: "", startDate: "", endDate: "" }
  });

  const updateFilter = (analysisId: string, field: keyof AnalysisFilters, value: string) => {
    setFilters(prev => ({
      ...prev,
      [analysisId]: {
        ...prev[analysisId],
        [field]: value
      }
    }));
  };

  const handleRunAnalysis = (analysisId: string) => {
    const filter = filters[analysisId];
    console.log(`Executando análise ${analysisId}:`, filter);
    // TODO: Integrar com backend/webhook
  };

  const isAnalysisReady = (analysisId: string) => {
    const filter = filters[analysisId];
    return filter.store && filter.startDate && filter.endDate;
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Análises</h1>
        <p className="text-muted-foreground">Selecione o tipo de análise e configure os filtros</p>
      </div>

      {/* Analysis Cards */}
      <div className="grid gap-6 md:grid-cols-1 lg:grid-cols-3">
        {analysisTypes.map((analysis) => (
          <Card 
            key={analysis.id} 
            className={`${analysis.bgColor} ${analysis.borderColor} border-2`}
          >
            <CardHeader>
              <CardTitle className={`flex items-center gap-2 ${analysis.color}`}>
                <Search className="h-5 w-5" />
                {analysis.title}
              </CardTitle>
              <CardDescription className="text-muted-foreground text-sm">
                {analysis.description}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Store Input */}
              <div className="space-y-2">
                <Label className="text-foreground text-sm">Loja</Label>
                <div className="relative">
                  <Store className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    type="text"
                    placeholder="Digite o número da loja"
                    value={filters[analysis.id].store}
                    onChange={(e) => updateFilter(analysis.id, "store", e.target.value)}
                    className="pl-9 bg-background border-border"
                  />
                </div>
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label className="text-foreground text-sm">Data Inicial</Label>
                  <div className="relative">
                    <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="date"
                      value={filters[analysis.id].startDate}
                      onChange={(e) => updateFilter(analysis.id, "startDate", e.target.value)}
                      className="pl-9 bg-background border-border"
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label className="text-foreground text-sm">Data Final</Label>
                  <div className="relative">
                    <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      type="date"
                      value={filters[analysis.id].endDate}
                      onChange={(e) => updateFilter(analysis.id, "endDate", e.target.value)}
                      className="pl-9 bg-background border-border"
                    />
                  </div>
                </div>
              </div>

              {/* Run Button */}
              <Button 
                className="w-full mt-2" 
                disabled={!isAnalysisReady(analysis.id)}
                onClick={() => handleRunAnalysis(analysis.id)}
              >
                <Play className="h-4 w-4 mr-2" />
                Executar Análise
              </Button>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
