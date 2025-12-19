import { cn } from "@/lib/utils";

interface PriorityBadgeProps {
  prioridade: 'baixo' | 'médio' | 'crítico';
  className?: string;
}

const prioridadeConfig = {
  baixo: {
    label: 'Baixo',
    className: 'bg-blue-500/10 text-blue-500 border-blue-500/20'
  },
  'médio': {
    label: 'Médio',
    className: 'bg-amber-500/10 text-amber-500 border-amber-500/20'
  },
  crítico: {
    label: 'Crítico',
    className: 'bg-red-500/10 text-red-500 border-red-500/20'
  }
};

export function PriorityBadge({ prioridade, className }: PriorityBadgeProps) {
  const config = prioridadeConfig[prioridade] || prioridadeConfig['baixo'];
  
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
