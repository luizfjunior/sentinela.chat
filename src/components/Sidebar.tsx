
import { useState } from "react";
import { Menu, Plus, MoreHorizontal, Trash2, Edit3 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { RenameConversationDialog } from "./RenameConversationDialog";
import { Conversation } from "@/hooks/useSupabaseConversations";

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  conversations: Conversation[];
  currentConversationId: string | null;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
}

const Sidebar = ({
  isOpen,
  onToggle,
  conversations,
  currentConversationId,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onRenameConversation
}: SidebarProps) => {
  return <>
      {/* Toggle Button - Melhorado */}
      <Button 
        variant="ghost" 
        size="icon" 
        onClick={onToggle} 
        className="fixed top-4 left-4 z-50 text-white bg-zinc-700 hover:bg-zinc-600 h-10 w-10 flex items-center justify-center rounded-lg transition-all duration-200"
      >
        <Menu className="h-5 w-5" />
      </Button>

      {/* Sidebar */}
      <div className={cn("fixed left-0 top-0 h-full bg-zinc-900 border-r border-zinc-800 transition-transform duration-300 z-40", isOpen ? "translate-x-0" : "-translate-x-full", "w-64")}>
        {/* Header */}
        <div className="p-4 border-b border-zinc-800">
          <h1 className="tracking-wider text-right font-extrabold text-2xl text-zinc-400">SENTINELA</h1>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          <Button onClick={onNewChat} className="w-full justify-start gap-2 bg-zinc-800 hover:bg-zinc-700 text-white">
            <Plus className="h-4 w-4" />
            Novo chat
          </Button>
        </div>

        {/* Conversations List */}
        <div className="flex-1 overflow-y-auto">
          {conversations.map(conversation => (
            <div 
              key={conversation.id} 
              className={cn(
                "group flex items-center justify-between px-4 py-2 mx-2 rounded-lg cursor-pointer hover:bg-zinc-800 transition-colors", 
                currentConversationId === conversation.id && "bg-zinc-800"
              )} 
              onClick={() => onSelectConversation(conversation.id)}
            >
              <span className="text-white text-sm truncate flex-1">
                {conversation.title}
              </span>
              
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="opacity-0 group-hover:opacity-100 transition-opacity h-6 w-6 p-0 text-zinc-400 hover:text-white" 
                    onClick={e => e.stopPropagation()}
                  >
                    <MoreHorizontal className="h-3 w-3" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="bg-zinc-800 border-zinc-700">
                  <RenameConversationDialog
                    currentTitle={conversation.title}
                    onRename={(newTitle) => onRenameConversation(conversation.id, newTitle)}
                  >
                    <DropdownMenuItem 
                      onSelect={(e) => e.preventDefault()}
                      className="text-blue-400 hover:text-blue-300 hover:bg-zinc-700 cursor-pointer"
                    >
                      <Edit3 className="h-4 w-4 mr-2" />
                      Alterar nome
                    </DropdownMenuItem>
                  </RenameConversationDialog>
                  <DropdownMenuItem 
                    onClick={e => {
                      e.stopPropagation();
                      onDeleteConversation(conversation.id);
                    }} 
                    className="text-red-400 hover:text-red-300 hover:bg-zinc-700 cursor-pointer"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Excluir conversa
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          ))}
        </div>
      </div>

      {/* Overlay */}
      {isOpen && <div className="fixed inset-0 bg-black/20 z-30" onClick={onToggle} />}
    </>;
};

export default Sidebar;
