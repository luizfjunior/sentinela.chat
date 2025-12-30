import { useState } from "react";
import { User, Bell, Shield, Store, Save } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { mockStores } from "@/data/mockData";
import { Checkbox } from "@/components/ui/checkbox";

export default function Configuracoes() {
  const { user, profile } = useAuth();
  const [notifyEmail, setNotifyEmail] = useState(true);
  const [notifyCritical, setNotifyCritical] = useState(true);
  const [notifyMedium, setNotifyMedium] = useState(true);
  const [notifyLight, setNotifyLight] = useState(false);
  const [selectedStores, setSelectedStores] = useState<string[]>(mockStores.map(s => s.id));

  const handleSave = () => {
    toast.success("Configurações salvas com sucesso!");
  };

  const toggleStore = (storeId: string) => {
    setSelectedStores(prev => 
      prev.includes(storeId) 
        ? prev.filter(id => id !== storeId)
        : [...prev, storeId]
    );
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Configurações</h1>
        <p className="text-muted-foreground">Gerencie suas preferências e configurações do sistema</p>
      </div>

      <Tabs defaultValue="perfil" className="space-y-6">
        <TabsList className="bg-secondary">
          <TabsTrigger value="perfil" className="flex items-center gap-2">
            <User className="h-4 w-4" />
            Perfil
          </TabsTrigger>
          <TabsTrigger value="notificacoes" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            Notificações
          </TabsTrigger>
          <TabsTrigger value="alertas" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            Regras de Alerta
          </TabsTrigger>
          <TabsTrigger value="lojas" className="flex items-center gap-2">
            <Store className="h-4 w-4" />
            Lojas
          </TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="perfil">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle>Informações do Perfil</CardTitle>
              <CardDescription>Gerencie suas informações pessoais</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label className="text-muted-foreground">Nome Completo</Label>
                  <Input 
                    defaultValue={profile?.full_name || ""} 
                    className="bg-background border-border"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-muted-foreground">Email</Label>
                  <Input 
                    defaultValue={user?.email || ""} 
                    disabled 
                    className="bg-background border-border"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-muted-foreground">Cargo</Label>
                  <Input 
                    defaultValue="Auditor" 
                    className="bg-background border-border"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-muted-foreground">Departamento</Label>
                  <Input 
                    defaultValue="Auditoria Interna" 
                    className="bg-background border-border"
                  />
                </div>
              </div>
              <Separator className="bg-border" />
              <Button onClick={handleSave}>
                <Save className="h-4 w-4 mr-2" />
                Salvar Alterações
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notifications Tab */}
        <TabsContent value="notificacoes">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle>Preferências de Notificação</CardTitle>
              <CardDescription>Configure como deseja receber alertas</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label className="text-foreground">Notificações por Email</Label>
                    <p className="text-sm text-muted-foreground">
                      Receber alertas no seu email
                    </p>
                  </div>
                  <Switch checked={notifyEmail} onCheckedChange={setNotifyEmail} />
                </div>
                <Separator className="bg-border" />
                <div className="space-y-4">
                  <Label className="text-foreground">Níveis de Alerta</Label>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-3 h-3 rounded-full bg-red-500" />
                        <span className="text-sm text-foreground">Alertas Críticos</span>
                      </div>
                      <Switch checked={notifyCritical} onCheckedChange={setNotifyCritical} />
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-3 h-3 rounded-full bg-amber-500" />
                        <span className="text-sm text-foreground">Alertas Médios</span>
                      </div>
                      <Switch checked={notifyMedium} onCheckedChange={setNotifyMedium} />
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-3 h-3 rounded-full bg-emerald-500" />
                        <span className="text-sm text-foreground">Alertas Leves</span>
                      </div>
                      <Switch checked={notifyLight} onCheckedChange={setNotifyLight} />
                    </div>
                  </div>
                </div>
              </div>
              <Separator className="bg-border" />
              <Button onClick={handleSave}>
                <Save className="h-4 w-4 mr-2" />
                Salvar Preferências
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alert Rules Tab */}
        <TabsContent value="alertas">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle>Regras de Classificação</CardTitle>
              <CardDescription>Configure os critérios para classificação de alertas</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg">
                  <h4 className="font-medium text-red-500 mb-2">🔴 CRÍTICO</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Diferença para pagar até R$ 10,00 ou negativa</li>
                    <li>• Entrada: mais de 1 item | Saída: até 10 itens</li>
                    <li>• Item devolvido é Meias → item retirado é Calçados</li>
                    <li>• Comprador ≠ pessoa que faz a troca</li>
                    <li>• Produto devolvido {'>'} 90 dias da compra</li>
                  </ul>
                </div>
                <div className="p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
                  <h4 className="font-medium text-amber-500 mb-2">🟡 MÉDIO</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Trocas entre marcas/departamentos diferentes</li>
                    <li>• Sem diferença ou com diferença negativa</li>
                    <li>• Entrada: 1 item | Saída: até 3 itens</li>
                  </ul>
                </div>
                <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                  <h4 className="font-medium text-emerald-500 mb-2">🟢 LEVE</h4>
                  <ul className="text-sm text-muted-foreground space-y-1">
                    <li>• Trocas da mesma marca e mesmo departamento</li>
                    <li>• Com diferença para pagar</li>
                    <li>• Entrada: 1 item | Saída: 1 item</li>
                  </ul>
                </div>
              </div>
              <Separator className="bg-border" />
              <p className="text-sm text-muted-foreground">
                As regras de classificação são gerenciadas pelo sistema. Entre em contato com o administrador para alterações.
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Stores Tab */}
        <TabsContent value="lojas">
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle>Lojas Monitoradas</CardTitle>
              <CardDescription>Selecione as lojas que deseja monitorar</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {mockStores.map((store) => (
                  <div 
                    key={store.id}
                    className="flex items-center space-x-3 p-4 bg-background border border-border rounded-lg"
                  >
                    <Checkbox 
                      id={store.id}
                      checked={selectedStores.includes(store.id)}
                      onCheckedChange={() => toggleStore(store.id)}
                    />
                    <div className="flex-1">
                      <label 
                        htmlFor={store.id} 
                        className="text-sm font-medium text-foreground cursor-pointer"
                      >
                        {store.nome}
                      </label>
                      <p className="text-xs text-muted-foreground">
                        {store.alertasHoje} alertas hoje
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              <Separator className="bg-border" />
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  {selectedStores.length} de {mockStores.length} lojas selecionadas
                </span>
                <Button onClick={handleSave}>
                  <Save className="h-4 w-4 mr-2" />
                  Salvar Seleção
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
