import { cn } from "@/lib/utils";

interface PriorityBadgeProps {
  prioridade: 'baixa' | 'media' | 'alta' | 'urgente';
  className?: string;
}

const prioridadeConfig = {
  baixa: {
    label: 'Baixa',
    className: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
  },
  media: {
    label: 'Média',
    className: 'bg-blue-500/10 text-blue-500 border-blue-500/20'
  },
  alta: {
    label: 'Alta',
    className: 'bg-amber-500/10 text-amber-500 border-amber-500/20'
  },
  urgente: {
    label: 'Urgente',
    className: 'bg-red-500/10 text-red-500 border-red-500/20'
  }
};

export function PriorityBadge({ prioridade, className }: PriorityBadgeProps) {
  const config = prioridadeConfig[prioridade];
  
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
