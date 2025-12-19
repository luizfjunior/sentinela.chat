import { cn } from "@/lib/utils";

interface StatusBadgeProps {
  status: string;
  className?: string;
}

const statusConfig: Record<string, { label: string; className: string }> = {
  pendente: {
    label: 'Ocorrência',
    className: 'bg-amber-500/10 text-amber-500 border-amber-500/20'
  },
  em_analise: {
    label: 'Em Análise',
    className: 'bg-blue-500/10 text-blue-500 border-blue-500/20'
  },
  resolvido: {
    label: 'Ocorrência',
    className: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
  },
  falso_positivo: {
    label: 'Falso Positivo',
    className: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
  },
  aberto: {
    label: 'Ocorrência',
    className: 'bg-blue-500/10 text-blue-500 border-blue-500/20'
  },
  pendente_resposta: {
    label: 'Não Fraude',
    className: 'bg-purple-500/10 text-purple-500 border-purple-500/20'
  },
  concluido: {
    label: 'Fraude',
    className: 'bg-red-500/10 text-red-500 border-red-500/20'
  },
  arquivado: {
    label: 'Não Fraude',
    className: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
  }
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const config = statusConfig[status] || {
    label: status,
    className: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
  };
  
  return (
    <span className={cn(
      "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border",
      config.className,
      className
    )}>
      {config.label}
    </span>
  );
}
