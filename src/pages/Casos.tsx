import { useState } from "react";
import { Plus, Search, LayoutGrid, Calendar, Store } from "lucide-react";
import { mockCases, Case } from "@/data/mockData";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { PriorityBadge } from "@/components/shared/PriorityBadge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { format } from "date-fns";
import { ptBR } from "date-fns/locale";
import { DragDropContext, Droppable, Draggable, DropResult } from "@hello-pangea/dnd";
import { toast } from "sonner";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

export default function Casos() {
  const [cases, setCases] = useState<Case[]>([...mockCases]);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedStatus, setSelectedStatus] = useState<string>("all");
  const [selectedPriority, setSelectedPriority] = useState<string>("all");
  const [viewMode, setViewMode] = useState<"kanban" | "list">("kanban");
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // Form state
  const [novaOcorrencia, setNovaOcorrencia] = useState({
    titulo: "",
    descricao: "",
    loja: "",
    prioridade: "médio" as Case['prioridade'],
    valorEnvolvido: ""
  });

  const handleCreateOcorrencia = () => {
    if (!novaOcorrencia.titulo.trim()) {
      toast.error("Título é obrigatório");
      return;
    }
    if (!novaOcorrencia.loja.trim()) {
      toast.error("Loja é obrigatória");
      return;
    }

    const newCase: Case = {
      id: `case-${Date.now()}`,
      titulo: novaOcorrencia.titulo,
      descricao: novaOcorrencia.descricao,
      status: "aberto",
      prioridade: novaOcorrencia.prioridade,
      lojaId: novaOcorrencia.loja,
      lojaNome: `Loja ${novaOcorrencia.loja}`,
      alertasVinculados: 0,
      valorTotalEnvolvido: parseFloat(novaOcorrencia.valorEnvolvido) || 0,
      responsavel: "Não atribuído",
      createdAt: new Date()
    };

    setCases(prev => [newCase, ...prev]);
    setNovaOcorrencia({
      titulo: "",
      descricao: "",
      loja: "",
      prioridade: "médio",
      valorEnvolvido: ""
    });
    setIsModalOpen(false);
    toast.success("Ocorrência criada com sucesso!");
  };
  const filteredCases = cases.filter(caso => {
    const matchesSearch = caso.titulo.toLowerCase().includes(searchTerm.toLowerCase()) || caso.descricao.toLowerCase().includes(searchTerm.toLowerCase()) || caso.lojaNome.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus = selectedStatus === "all" || caso.status === selectedStatus;
    const matchesPriority = selectedPriority === "all" || caso.prioridade === selectedPriority;
    return matchesSearch && matchesStatus && matchesPriority;
  });
  const kanbanColumns = [{
    id: 'aberto',
    title: 'Ocorrência',
    cases: filteredCases.filter(c => c.status === 'aberto')
  }, {
    id: 'em_analise',
    title: 'Em Análise',
    cases: filteredCases.filter(c => c.status === 'em_analise')
  }, {
    id: 'pendente_resposta',
    title: 'Não Fraude',
    cases: filteredCases.filter(c => c.status === 'pendente_resposta')
  }, {
    id: 'concluido',
    title: 'Fraude',
    cases: filteredCases.filter(c => c.status === 'concluido' || c.status === 'arquivado')
  }];
  const statusLabels: Record<string, string> = {
    'aberto': 'Ocorrência',
    'em_analise': 'Em Análise',
    'pendente_resposta': 'Não Fraude',
    'concluido': 'Fraude'
  };
  const formatDate = (date: Date) => {
    return format(date, "dd/MM/yyyy", {
      locale: ptBR
    });
  };
  const handleDragEnd = (result: DropResult) => {
    const {
      destination,
      source,
      draggableId
    } = result;

    // Se não houver destino ou for o mesmo lugar, não faz nada
    if (!destination) return;
    if (destination.droppableId === source.droppableId && destination.index === source.index) return;

    // Atualiza o status do caso
    const newStatus = destination.droppableId as Case['status'];
    setCases(prevCases => prevCases.map(caso => caso.id === draggableId ? {
      ...caso,
      status: newStatus
    } : caso));
    toast.success(`Caso movido para "${statusLabels[newStatus]}"`);
  };
  const CaseCard = ({
    caso,
    index
  }: {
    caso: Case;
    index: number;
  }) => <Draggable draggableId={caso.id} index={index}>
      {(provided, snapshot) => <div ref={provided.innerRef} {...provided.draggableProps} {...provided.dragHandleProps}>
          <Card className={`bg-background border-border hover:border-primary/50 transition-all cursor-grab active:cursor-grabbing ${snapshot.isDragging ? 'shadow-lg ring-2 ring-primary/50 rotate-2' : ''}`}>
            <CardContent className="p-4 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-sm font-medium text-foreground line-clamp-2">{caso.titulo}</h3>
                <PriorityBadge prioridade={caso.prioridade} />
              </div>
              <p className="text-xs text-muted-foreground line-clamp-2">{caso.descricao}</p>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                
                <span className="flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {formatDate(caso.createdAt)}
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-border">
                <span className="text-xs text-muted-foreground">{caso.lojaNome}</span>
                <span className="text-xs font-medium text-primary">
                  R$ {caso.valorTotalEnvolvido.toFixed(2)}
                </span>
              </div>
              
            </CardContent>
          </Card>
        </div>}
    </Draggable>;
  return <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Ocorrências</h1>
          <p className="text-muted-foreground">Gerencie suas investigações e riscos de fraude</p>
        </div>
        <Button className="w-fit" onClick={() => setIsModalOpen(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Nova Ocorrência   
        </Button>
      </div>

      {/* Modal Nova Ocorrência */}
      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Nova Ocorrência</DialogTitle>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="titulo">Título *</Label>
              <Input
                id="titulo"
                placeholder="Ex: Suspeita de fraude em devolução"
                value={novaOcorrencia.titulo}
                onChange={(e) => setNovaOcorrencia(prev => ({ ...prev, titulo: e.target.value }))}
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="descricao">Descrição</Label>
              <Textarea
                id="descricao"
                placeholder="Descreva os detalhes da ocorrência..."
                rows={3}
                value={novaOcorrencia.descricao}
                onChange={(e) => setNovaOcorrencia(prev => ({ ...prev, descricao: e.target.value }))}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="loja">Loja *</Label>
                <div className="relative">
                  <Store className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="loja"
                    placeholder="Número da loja"
                    className="pl-9"
                    value={novaOcorrencia.loja}
                    onChange={(e) => setNovaOcorrencia(prev => ({ ...prev, loja: e.target.value }))}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="prioridade">Prioridade</Label>
                <Select
                  value={novaOcorrencia.prioridade}
                  onValueChange={(value: Case['prioridade']) => setNovaOcorrencia(prev => ({ ...prev, prioridade: value }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="baixo">Baixo</SelectItem>
                    <SelectItem value="médio">Médio</SelectItem>
                    <SelectItem value="crítico">Crítico</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="valor">Valor Envolvido (R$)</Label>
              <Input
                id="valor"
                type="number"
                placeholder="0,00"
                value={novaOcorrencia.valorEnvolvido}
                onChange={(e) => setNovaOcorrencia(prev => ({ ...prev, valorEnvolvido: e.target.value }))}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsModalOpen(false)}>
              Cancelar
            </Button>
            <Button onClick={handleCreateOcorrencia}>
              Criar Ocorrência
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Filters */}
      <Card className="bg-card border-border">
        <CardContent className="pt-4">
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input placeholder="Buscar casos..." value={searchTerm} onChange={e => setSearchTerm(e.target.value)} className="pl-9 bg-background border-border" />
            </div>
            <Select value={selectedStatus} onValueChange={setSelectedStatus}>
              <SelectTrigger className="w-full sm:w-[180px] bg-background border-border">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Todos os Status</SelectItem>
                <SelectItem value="aberto">Ocorrência</SelectItem>
                <SelectItem value="em_analise">Em Análise</SelectItem>
                <SelectItem value="pendente_resposta">Não Fraude</SelectItem>
                <SelectItem value="concluido">Fraude</SelectItem>
              </SelectContent>
            </Select>
            <Select value={selectedPriority} onValueChange={setSelectedPriority}>
              <SelectTrigger className="w-full sm:w-[180px] bg-background border-border">
                <SelectValue placeholder="Prioridade" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Todas</SelectItem>
                <SelectItem value="crítico">Crítico</SelectItem>
                <SelectItem value="médio">Médio</SelectItem>
                <SelectItem value="baixo">Baixo</SelectItem>
              </SelectContent>
            </Select>
            <div className="flex gap-1 border border-border rounded-md p-1">
              <Button variant={viewMode === "kanban" ? "secondary" : "ghost"} size="icon" className="h-8 w-8" onClick={() => setViewMode("kanban")}>
                <LayoutGrid className="h-4 w-4" />
              </Button>
              
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Kanban View with Drag and Drop */}
      {viewMode === "kanban" && <DragDropContext onDragEnd={handleDragEnd}>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {kanbanColumns.map(column => <div key={column.id} className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium text-foreground">{column.title}</h3>
                  <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded">
                    {column.cases.length}
                  </span>
                </div>
                <Droppable droppableId={column.id}>
                  {(provided, snapshot) => <div ref={provided.innerRef} {...provided.droppableProps} className={`space-y-3 min-h-[200px] rounded-lg p-2 transition-colors ${snapshot.isDraggingOver ? 'bg-primary/10 ring-2 ring-primary/30' : 'bg-secondary/30'}`}>
                      {column.cases.map((caso, index) => <CaseCard key={caso.id} caso={caso} index={index} />)}
                      {provided.placeholder}
                      {column.cases.length === 0 && !snapshot.isDraggingOver && <div className="text-center py-8 text-muted-foreground text-sm">
                          Nenhum caso
                        </div>}
                    </div>}
                </Droppable>
              </div>)}
          </div>
        </DragDropContext>}

      {/* List View */}
      {viewMode === "list" && <div className="space-y-3">
          {filteredCases.map(caso => <Card key={caso.id} className="bg-card border-border hover:border-primary/50 transition-colors cursor-pointer">
              <CardContent className="p-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-3">
                      <h3 className="font-medium text-foreground">{caso.titulo}</h3>
                      <PriorityBadge prioridade={caso.prioridade} />
                      <StatusBadge status={caso.status} />
                    </div>
                    <p className="text-sm text-muted-foreground">{caso.descricao}</p>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <span>{caso.lojaNome}</span>
                      <span>{caso.alertasVinculados} alertas vinculados</span>
                      <span>Criado em {formatDate(caso.createdAt)}</span>
                      <span>Responsável: {caso.responsavel}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-lg font-bold text-primary">
                      R$ {caso.valorTotalEnvolvido.toFixed(2)}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>)}
          {filteredCases.length === 0 && <div className="py-12 text-center text-muted-foreground">
              Nenhum caso encontrado com os filtros selecionados
            </div>}
        </div>}
    </div>;
}