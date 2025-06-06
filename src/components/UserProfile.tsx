
import { useState } from "react";
import { User, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useAuth } from "@/contexts/AuthContext";

const UserProfile = () => {
  const { user, profile, signOut } = useAuth();
  
  const userEmail = user?.email || profile?.email || "usuario@grupooscar.com";
  const userName = profile?.full_name || user?.user_metadata?.full_name || userEmail.split('@')[0];
  const initials = userName.substring(0, 2).toUpperCase();

  const handleLogout = async () => {
    await signOut();
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" className="h-8 w-8 rounded-full p-0">
          <Avatar className="h-8 w-8">
            <AvatarFallback className="bg-zinc-600 text-white text-xs">
              {initials}
            </AvatarFallback>
          </Avatar>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56 bg-zinc-800 border-zinc-700">
        <div className="px-3 py-2">
          <p className="text-sm text-white font-medium">{userName}</p>
          <p className="text-xs text-zinc-400">{userEmail}</p>
        </div>
        <DropdownMenuSeparator className="bg-zinc-700" />
        <DropdownMenuItem
          onClick={handleLogout}
          className="text-white hover:bg-zinc-700 cursor-pointer"
        >
          <LogOut className="h-4 w-4 mr-2" />
          Terminar Sessão
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default UserProfile;
