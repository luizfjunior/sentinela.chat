import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  MessageSquare,
  Search,
  FolderOpen,
  ChevronLeft,
  ChevronRight,
  UserCog,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import UserProfile from "@/components/UserProfile";

interface AppSidebarProps {
  isCollapsed: boolean;
  onToggle: () => void;
}

const navItems = [
  {
    title: "Dashboard",
    url: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "Chat IA",
    url: "/chat",
    icon: MessageSquare,
  },
  {
    title: "Análises",
    url: "/alertas",
    icon: Search,
  },
  {
    title: "Ocorrências",
    url: "/casos",
    icon: FolderOpen,
  },
  {
    title: "Admin",
    url: "/admin",
    icon: UserCog,
  },
];

export function AppSidebar({ isCollapsed, onToggle }: AppSidebarProps) {
  const location = useLocation();

  return (
    <aside
      className={cn(
        "h-screen bg-card border-r border-border flex flex-col transition-all duration-300 ease-in-out",
        isCollapsed ? "w-16" : "w-64"
      )}
    >
      {/* Header */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-border">
        {!isCollapsed && (
          <span className="font-bold text-lg text-foreground tracking-wider">
            SENTINELA
          </span>
        )}

        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className={cn("h-8 w-8", isCollapsed && "mx-auto")}
        >
          {isCollapsed ? (
            <ChevronRight className="h-4 w-4" />
          ) : (
            <ChevronLeft className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-2 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = location.pathname === item.url;

          return (
            <NavLink
              key={item.url}
              to={item.url}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors",
                "hover:bg-accent hover:text-accent-foreground",
                isActive && "bg-primary/10 text-primary font-medium",
                isCollapsed && "justify-center px-2"
              )}
            >
              <item.icon
                className={cn(
                  "h-5 w-5 flex-shrink-0",
                  isActive && "text-primary"
                )}
              />

              {!isCollapsed && (
                <span className="flex-1">{item.title}</span>
              )}
            </NavLink>
          );
        })}
      </nav>

      {/* Footer - User Profile */}
      <div
        className={cn(
          "border-t border-border p-3",
          isCollapsed && "flex justify-center"
        )}
      >
        <UserProfile collapsed={isCollapsed} />
      </div>
    </aside>
  );
}
