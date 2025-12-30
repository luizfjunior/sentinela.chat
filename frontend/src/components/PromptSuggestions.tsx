import { FileSearch, TrendingUp, AlertTriangle } from "lucide-react";

export interface PromptSuggestion {
  id: string;
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  prompt: string;
  category?: string;
  description: string;
}

export const suggestions: PromptSuggestion[] = [
  {
    id: "saida-troca",
    title: "Análise de Saída/Troca",
    icon: FileSearch,
    description: "Identifica produtos com saída por ajuste/inventário e devoluções",
    prompt: `Solicito uma análise, que apresente todos os produtos da Loja 111 que cumpriram as seguintes condições, no período de Janeiro a Junho de 2025:
Condições para Análise:
O produto deve ter tido qualquer tipo de saída por inventário ou ajuste de contagem.
O mesmo produto (pelo código/SKU) deve também aparecer em qualquer registro de devolução por troca no mesmo período. 
E me traga todas as evidencias de datas referente ao pedido acima e me traga aqui para uma conferencia.`,
    category: "Análise"
  }
];

interface PromptSuggestionsProps {
  onSuggestionClick: (prompt: string) => void;
  isVisible: boolean;
}

const PromptSuggestions = ({ onSuggestionClick, isVisible }: PromptSuggestionsProps) => {
  if (!isVisible) return null;

  return (
    <div className="absolute bottom-full mb-3 w-full animate-fade-in animate-scale-in">
      <div className="flex flex-col sm:flex-row gap-3">
        {suggestions.map((suggestion) => {
          const Icon = suggestion.icon;
          return (
            <button
              key={suggestion.id}
              onClick={() => onSuggestionClick(suggestion.prompt)}
              className="flex-1 group relative overflow-hidden rounded-xl border border-gray-700 bg-gray-800/50 backdrop-blur-sm p-4 text-left transition-all duration-200 hover:bg-gray-700/70 hover:border-blue-500 hover:shadow-lg hover:scale-[1.02]"
            >
              <div className="flex items-start gap-3">
                <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center group-hover:bg-blue-500/20 transition-colors">
                  <Icon className="h-5 w-5 text-blue-400" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-semibold text-white mb-1">
                    {suggestion.title}
                  </h3>
                  <p className="text-xs text-gray-400 line-clamp-2">
                    {suggestion.description}
                  </p>
                  {suggestion.category && (
                    <span className="inline-block mt-2 text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-300">
                      {suggestion.category}
                    </span>
                  )}
                </div>
              </div>
              
              {/* Hover effect overlay */}
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500/0 via-blue-500/5 to-blue-500/0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
            </button>
          );
        })}
      </div>
      
      {/* Helper text */}
      <p className="text-xs text-gray-500 text-center mt-2">
        💡 Clique em uma sugestão para preencher o prompt e editar conforme necessário
      </p>
    </div>
  );
};

export default PromptSuggestions;
