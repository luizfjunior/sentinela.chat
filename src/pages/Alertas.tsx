import { useState } from "react";
import { Search, Play, CalendarIcon } from "lucide-react";
import { mockStores } from "@/data/mockData";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { format } from "date-fns";
import { ptBR } from "date-fns/locale";
import { cn } from "@/lib/utils";

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
  startDate: Date | undefined;
  endDate: Date | undefined;
}

export default function Alertas() {
  const [filters, setFilters] = useState<Record<string, AnalysisFilters>>({
    leve: { store: "", startDate: undefined, endDate: undefined },
    medio: { store: "", startDate: undefined, endDate: undefined },
    critico: { store: "", startDate: undefined, endDate: undefined }
  });

  const updateFilter = (analysisId: string, field: keyof AnalysisFilters, value: string | Date | undefined) => {
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
              {/* Store Select */}
              <div className="space-y-2">
                <Label className="text-foreground text-sm">Loja</Label>
                <Select 
                  value={filters[analysis.id].store} 
                  onValueChange={(value) => updateFilter(analysis.id, "store", value)}
                >
                  <SelectTrigger className="bg-background border-border">
                    <SelectValue placeholder="Selecione a loja" />
                  </SelectTrigger>
                  <SelectContent>
                    {mockStores.map(store => (
                      <SelectItem key={store.id} value={store.id}>{store.nome}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label className="text-foreground text-sm">Data Inicial</Label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className={cn(
                          "w-full justify-start text-left font-normal bg-background border-border",
                          !filters[analysis.id].startDate && "text-muted-foreground"
                        )}
                      >
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        {filters[analysis.id].startDate 
                          ? format(filters[analysis.id].startDate, "dd/MM/yyyy", { locale: ptBR })
                          : <span>Selecione</span>
                        }
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={filters[analysis.id].startDate}
                        onSelect={(date) => updateFilter(analysis.id, "startDate", date)}
                        initialFocus
                        className={cn("p-3 pointer-events-auto")}
                      />
                    </PopoverContent>
                  </Popover>
                </div>
                <div className="space-y-2">
                  <Label className="text-foreground text-sm">Data Final</Label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className={cn(
                          "w-full justify-start text-left font-normal bg-background border-border",
                          !filters[analysis.id].endDate && "text-muted-foreground"
                        )}
                      >
                        <CalendarIcon className="mr-2 h-4 w-4" />
                        {filters[analysis.id].endDate 
                          ? format(filters[analysis.id].endDate, "dd/MM/yyyy", { locale: ptBR })
                          : <span>Selecione</span>
                        }
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={filters[analysis.id].endDate}
                        onSelect={(date) => updateFilter(analysis.id, "endDate", date)}
                        initialFocus
                        className={cn("p-3 pointer-events-auto")}
                      />
                    </PopoverContent>
                  </Popover>
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
