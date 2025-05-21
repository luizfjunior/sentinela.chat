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
        <h3 className="text-blue-400 font-medium mb-2">🔹Exemplo de um possível processo realizado e estruturado: </h3>
        
        <p><span className="highlighted-text">Processo:</span> 1 - Planejamento Comercial de Coleções (Tático)</p>
        <p><span className="highlighted-text">Entrada:</span> Calendário comercial e metas de vendas</p>
        
        <div className="ml-4 mb-3">
          <p><span className="highlighted-text">Subprocesso:</span> 1.1 Análise de Calendário Comercial (Tático)</p>
          <p><span className="highlighted-text">Tarefa:</span> 1.1.1 Avaliar datas sazonais e concorrência</p>
          <ul className="ml-2 space-y-1">
            <li>• Acessar o calendário comercial interno (arquivo do Google Sheets ou planejamento anual da diretoria comercial)</li>
            <li>• Identificar datas relevantes para o varejo (ex: Dia das Mães, Black Friday, Natal, Liquidações Sazonais)</li>
            <li>• Consultar dados históricos de vendas nas mesmas datas (relatórios no Power BI ou no ERP/VTEX)</li>
            <li>• Analisar campanhas da concorrência através de sites, newsletters e redes sociais dos players do mercado</li>
            <li>• Definir as campanhas prioritárias para a temporada com base em potencial de vendas e metas por canal</li>
            <li>• Atualizar o calendário com campanhas confirmadas, datas de execução e responsáveis pelas entregas</li>
            <li>• Compartilhar o calendário validado com as áreas envolvidas (VM, Produtos, Estúdio, Cadastro)</li>
          </ul>
        </div>
        
        <div className="ml-4 mb-3">
          <p><span className="highlighted-text">Subprocesso:</span> 1.2 Seleção de Produtos por Coleção (Tático)</p>
          <p><span className="highlighted-text">Tarefa:</span> 1.2.1 Definir produtos estratégicos para campanhas</p>
          <ul className="ml-2 space-y-1">
            <li>• Abrir a planilha de lançamentos ou mix de produtos da temporada</li>
            <li>• Aplicar filtros por status de lançamento recente (ex: produtos cadastrados nos últimos 30 dias no MEGA ou VTEX)</li>
            <li>• Acessar o sistema MEGA para consultar margem bruta (markup) de cada produto</li>
            <li>• Priorizar produtos com margem acima de X% (ex: 55%), ou com alto giro em campanhas anteriores</li>
            <li>• Verificar disponibilidade de estoque nos canais 1P e lojas físicas</li>
            <li>• Validar se o produto possui material fotográfico pronto (consultar planilha de cadastro unificado ou pasta no Drive)</li>
            <li>• Indicar os produtos aprovados com marcador na planilha (coluna: "Campanha Ativa = Sim")</li>
            <li>• Consolidar a lista de produtos e exportar para inclusão no briefing da campanha</li>
          </ul>
        </div>
        
        <p><span className="highlighted-text">Saída:</span> Campanhas comerciais planejadas e aprovadas, com briefing enviado para VM, Estúdio e Cadastro.</p>
        
        <p className="mt-3"><span className="highlighted-text">Ferramentas:</span></p>
        <ul className="ml-2">
          <li>Planilha Google Sheets - Planejamento de campanhas, organização de coleções e compartilhamento com outras áreas.</li>
          <li>ClickUp - Gestão de tarefas e acompanhamento de demandas</li>
          <li>ERP - Acesso a dados de produtos como SKU, EAN, grade de numeração, etc</li>
        </ul>
      </div>
    </div>;
};
export default ProcessGuide;