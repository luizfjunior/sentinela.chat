
import React from 'react';
const ProcessGuide = () => {
  return <div className="content-section text-gray-200">
      <p>
        Um processo é um conjunto de atividades organizadas para alcançar um objetivo específico dentro do departamento. 
        Ele pode envolver mais de uma pessoa e pode estar ligado a diferentes setores da empresa.
      </p>

      <div className="content-card my-4">
        <h3 className="text-blue-400 font-medium mb-2">🔹 Estrutura do processo:</h3>
        
        <p><span className="highlighted-text">Processo:</span> "nome do processo"</p>
        <p><span className="highlighted-text">Entrada:</span> "entrada da atividade do processo para ser realizado"</p>
        
        <div className="ml-4 mb-3">
          <p><span className="highlighted-text">Subprocesso:</span> "nome do subprocesso"</p>
          <p><span className="highlighted-text">Tarefa:</span> "atividade desenvolvida"</p>
          <p className="ml-2">"ação da tarefa (exemplo: clico ali, faço aquilo para resolver isso)"</p>
        </div>
        
        <div className="ml-4 mb-3">
          <p><span className="highlighted-text">Subprocesso:</span> "nome do subprocesso"</p>
          <p><span className="highlighted-text">Tarefa:</span> "atividade desenvolvida"</p>
          <p className="ml-2">"ação da tarefa (exemplo: clico ali, faço aquilo para resolver isso)"</p>
        </div>
        
        <p><span className="highlighted-text">Saída:</span> "saída da atividade do processo realizado"</p>
        
        <p><span className="highlighted-text">Ferramentas</span> (exemplo: sistema 01 para coletar dados e planilhas compartilhadas)</p>
      </div>

      <div className="content-card mt-6">
        <h3 className="text-blue-400 font-medium mb-2">🔹Exemplo de um possível processo realizado e estruturado: </h3>
        
        <p><span className="highlighted-text">Processo:</span> Acompanhamento Logístico de Pedidos E-commerce (Tático)</p>
        <p><span className="highlighted-text">Entrada:</span> Pedidos faturados na VTEX e informações de rastreio geradas pelas transportadoras</p>
        
        <div className="ml-4 mb-3">
          <p><span className="highlighted-text">Subprocesso:</span> Monitoramento Diário dos Pedidos em Transporte</p>
          <p><span className="highlighted-text">Tarefa:</span> 1.1 Verificar status dos pedidos com transportadoras</p>
          <ul className="ml-2 space-y-1">
            <li>• Acessar o painel de logística (ex: Intelipost, Mandae, Jadlog ou integração via planilha)</li>
            <li>• Filtrar os pedidos com status "Em transporte" e data de envio até o dia anterior</li>
            <li>• Validar se há movimentações recentes no rastreio</li>
            <li>• Identificar pedidos com atraso, sem movimentação ou com status indefinido</li>
            <li>• Exportar a lista de pedidos com risco logístico (ex: atraso, erro de rota, entrega não realizada)</li>
          </ul>
          
          <p><span className="highlighted-text">Tarefa:</span> 1.2 Atualizar relatório de pedidos críticos</p>
          <ul className="ml-2 space-y-1">
            <li>• Abrir a planilha compartilhada "Pedidos com Pendência Logística"</li>
            <li>• Inserir os pedidos identificados no monitoramento, com nome do cliente, status, prazo e SLA</li>
            <li>• Classificar o tipo de problema (ex: atraso, erro de destino, devolução)</li>
            <li>• Compartilhar relatório com SAC e time de atendimento para ações preventivas</li>
            <li>• Atualizar a coluna "status tratativa" conforme evolução do caso</li>
          </ul>
        </div>
        
        <div className="ml-4 mb-3">
          <p><span className="highlighted-text">Subprocesso:</span> Acionamento e Tratativas com Transportadoras</p>
          <p><span className="highlighted-text">Tarefa:</span> 2.1 Abrir chamados junto à transportadora</p>
          <ul className="ml-2 space-y-1">
            <li>• Acessar o portal ou canal de atendimento da transportadora</li>
            <li>• Abrir protocolo com número do pedido, rastreio e descrição do problema</li>
            <li>• Salvar o número do chamado na planilha de pendências e registrar data de abertura</li>
            <li>• Aguardar resposta e atualizar status da tratativa diariamente</li>
          </ul>
          
          <p><span className="highlighted-text">Tarefa:</span> 2.2 Solicitar priorização ou reentrega</p>
          <ul className="ml-2 space-y-1">
            <li>• Quando aplicável, solicitar via canal de atendimento a priorização de entrega</li>
            <li>• Nos casos de devolução por ausência, solicitar nova tentativa</li>
            <li>• Confirmar retorno da transportadora e atualizar o cliente via SAC ou CRM interno</li>
            <li>• Encerrar o caso somente após confirmação de entrega ou devolução finalizada</li>
          </ul>
        </div>
        
        <p><span className="highlighted-text">Saída:</span> Pedidos monitorados, tratativas logísticas registradas e pendências solucionadas com transportadoras</p>
        
        <p className="mt-3"><span className="highlighted-text">Ferramentas:</span></p>
        <ul className="ml-2">
          <li>Plataforma de gestão de transporte (Intelipost, Jadlog, Correios)</li>
          <li>Planilha "Pendências Logísticas"</li>
          <li>Slack ou e-mail interno para integração com SAC</li>
        </ul>
      </div>
    </div>;
};
export default ProcessGuide;
