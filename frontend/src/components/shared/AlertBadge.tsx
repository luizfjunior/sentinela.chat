import { cn } from "@/lib/utils";

interface AlertBadgeProps {
  nivel: 'leve' | 'medio' | 'critico';
  className?: string;
}

const nivelConfig = {
  leve: {
    label: 'Leve',
    className: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
  },
  medio: {
    label: 'Médio',
    className: 'bg-amber-500/10 text-amber-500 border-amber-500/20'
  },
  critico: {
    label: 'Crítico',
    className: 'bg-red-500/10 text-red-500 border-red-500/20'
  }
};

export function AlertBadge({ nivel, className }: AlertBadgeProps) {
  const config = nivelConfig[nivel];
  
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
