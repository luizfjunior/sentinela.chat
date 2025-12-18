import { LogOut, Settings } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useAuth } from "@/contexts/AuthContext";
import { cn } from "@/lib/utils";

interface UserProfileProps {
  collapsed?: boolean;
}

const UserProfile = ({ collapsed = false }: UserProfileProps) => {
  const { user, profile, signOut } = useAuth();
  const navigate = useNavigate();
  
  const userEmail = user?.email || profile?.email || "usuario@grupooscar.com";
  const userName = profile?.full_name || user?.user_metadata?.full_name || userEmail.split('@')[0];
  const initials = userName.substring(0, 2).toUpperCase();

  const handleLogout = async () => {
    await signOut();
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className={cn(
          "rounded-full p-0",
          collapsed ? "h-10 w-10" : "h-auto w-full justify-start gap-3 px-2 py-2"
        )}>
          <Avatar className={cn(collapsed ? "h-10 w-10" : "h-8 w-8")}>
            <AvatarFallback className="bg-primary/20 text-primary text-xs">
              {initials}
            </AvatarFallback>
          </Avatar>
          {!collapsed && (
            <div className="flex flex-col items-start text-left">
              <span className="text-sm font-medium text-foreground truncate max-w-[140px]">{userName}</span>
              <span className="text-xs text-muted-foreground truncate max-w-[140px]">{userEmail}</span>
            </div>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56 bg-card border-border">
        <div className="px-3 py-2">
          <p className="text-sm text-foreground font-medium">{userName}</p>
          <p className="text-xs text-muted-foreground">{userEmail}</p>
        </div>
        <DropdownMenuSeparator className="bg-border" />
        <DropdownMenuItem
          onClick={() => navigate("/configuracoes")}
          className="text-foreground hover:bg-accent cursor-pointer"
        >
          <Settings className="h-4 w-4 mr-2" />
          Configurações
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={handleLogout}
          className="text-foreground hover:bg-accent cursor-pointer"
        >
          <LogOut className="h-4 w-4 mr-2" />
          Terminar Sessão
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default UserProfile;
