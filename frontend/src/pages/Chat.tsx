import { useState, useRef, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useSupabaseConversations } from "@/hooks/useSupabaseConversations";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import { toast } from "sonner";
import { MoreVertical, Plus, MessageSquare, Trash2, Edit3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
  isPending?: boolean; // for messages not yet saved to Supabase
}

const Chat = () => {
  const { user, profile } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [lastGeneratedMessageId, setLastGeneratedMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [isSheetOpen, setIsSheetOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState<string>("");

  const {
    conversations,
    currentConversationId,
    messages: supabaseMessages,
    createNewConversation,
    deleteConversation,
    saveMessage,
    updateConversationTitle,
    setCurrentConversationId,
    loadConversations
  } = useSupabaseConversations();

  // Load messages from Supabase only when conversation changes
  useEffect(() => {
    if (currentConversationId && supabaseMessages.length > 0) {
      // Convert supabaseMessages to local Message shape
      setMessages(
        supabaseMessages.map(msg => ({
          id: msg.id,
          content: msg.content,
          role: msg.role as "user" | "assistant",
          timestamp: new Date(msg.created_at)
        }))
      );
    } else if (!currentConversationId) {
      setMessages([]);
    }
  }, [currentConversationId, supabaseMessages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleNewChat = () => {
    setCurrentConversationId(null);
    setMessages([]);
    setLastGeneratedMessageId(null);
    setIsSheetOpen(false);
  };

  const handleSelectConversation = (id: string) => {
    setCurrentConversationId(id);
    setMessages([]);
    setLastGeneratedMessageId(null);
    setIsSheetOpen(false);
  };

  const handleDeleteConversation = (id: string) => {
    deleteConversation(id);
  };

  const handleRenameConversation = async (id: string, newTitle: string) => {
    try {
      await updateConversationTitle(id, newTitle);
      toast.success("Nome da conversa alterado com sucesso");
      setEditingId(null);
      setEditingTitle("");
    } catch (error) {
      console.error("Error renaming conversation:", error);
      toast.error("Erro ao alterar nome da conversa");
    }
  };

  const handleStartEdit = (id: string, title: string) => {
    setEditingId(id);
    setEditingTitle(title);
  };

  const handleSaveEdit = () => {
    if (editingId && editingTitle.trim()) {
      handleRenameConversation(editingId, editingTitle.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSaveEdit();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setEditingId(null);
      setEditingTitle("");
    }
  };

  // Helper: add message to local messages state (at end)
  const appendMessage = (msg: Message) => {
    setMessages(prev => [...prev, msg]);
  };

  // Remove pending (unsaved) user/assistant messages after saved and reloaded from Supabase
  useEffect(() => {
    // If supabaseMessages were loaded, remove pending ones (by id match/role)
    if (currentConversationId && supabaseMessages.length > 0) {
      setMessages(prev =>
        prev.filter(
          m =>
            !m.isPending ||
            !(
              supabaseMessages.findIndex(supMsg =>
                supMsg.content === m.content &&
                supMsg.role === m.role &&
                Math.abs(new Date(supMsg.created_at).getTime() - m.timestamp.getTime()) < 5000 // within 5s
              ) !== -1
            )
        )
      );
    }
  }, [supabaseMessages, currentConversationId]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim() || !user) return;

    setLoading(true);

    // Create or get conversation
    let convId = currentConversationId;
    if (!convId) {
      convId = await createNewConversation(content);
      if (!convId) {
        setLoading(false);
        return;
      }
    }

    // 1. Add user message immediately (pending)
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content,
      role: "user",
      timestamp: new Date(),
      isPending: true
    };
    appendMessage(userMessage);

    // 2. Save user message to Supabase (fire and forget)
    saveMessage(convId, content, 'user').then(() => {
      // Will be cleaned by useEffect when supabaseMessages load
    });

    // 3. If first message, update conversation title
    if (supabaseMessages.length === 0 && messages.length === 0) {
      const title = content.split(' ').slice(0, 3).join(' ') + '...';
      updateConversationTitle(convId, title);
    }

    // 4. Call backend for AI response (s2 API)
    let assistantContent = "";
    try {
      const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content })
      });
      const data = await response.json();
      assistantContent = data.answer || "Resposta vazia do servidor.";
    } catch (err: any) {
      assistantContent = "Erro ao obter resposta da IA.";
      toast.error("Erro ao buscar resposta da IA");
      console.error("API Error:", err);
    }

    // 5. Add AI message with pending/animation
    const assistantMessageId = `assistant-${Date.now()}`;
    const assistantMessage: Message = {
      id: assistantMessageId,
      content: assistantContent,
      role: "assistant",
      timestamp: new Date(),
      isPending: true
    };
    appendMessage(assistantMessage);
    setLastGeneratedMessageId(assistantMessageId);

    // 6. Save assistant message to Supabase
    saveMessage(convId, assistantContent, 'assistant').then(() => {
      // Will be cleaned by useEffect when supabaseMessages load
    });

    setLoading(false);
  };

  const handleSendAudio = async (audioBlob: Blob) => {
    if (!user) return;

    setLoading(true);

    // Create or get conversation
    let convId = currentConversationId;
    if (!convId) {
      convId = await createNewConversation("🎵 Áudio enviado");
      if (!convId) {
        setLoading(false);
        return;
      }
    }

    const textMessage = "🎵 Áudio enviado";

    // User message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      content: textMessage,
      role: "user",
      timestamp: new Date(),
      isPending: true
    };
    appendMessage(userMessage);

    saveMessage(convId, textMessage, 'user').then(() => { });

    try {
      // Convert blob to base64 for API
      const reader = new FileReader();
      reader.onload = async () => {
        const base64Audio = reader.result as string;
        let assistantContent = "";

        try {
          const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
          const response = await fetch(`${apiUrl}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: "[Áudio enviado - transcrição não suportada ainda]" })
          });
          const data = await response.json();
          assistantContent = data.answer || "Áudio processado.";
        } catch (err: any) {
          assistantContent = "Erro ao processar o áudio.";
          console.error("Audio API Error:", err);
        }

        const assistantMessageId = `assistant-${Date.now()}`;
        const assistantMessage: Message = {
          id: assistantMessageId,
          content: assistantContent,
          role: "assistant",
          timestamp: new Date(),
          isPending: true
        };
        appendMessage(assistantMessage);
        setLastGeneratedMessageId(assistantMessageId);

        saveMessage(convId!, assistantContent, 'assistant').then(() => { });
      };
      reader.readAsDataURL(audioBlob);
    } catch (error) {
      toast.error("Falha ao enviar áudio. Tente novamente.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header with conversations menu */}
      <div className="flex items-center justify-end px-4 py-2 border-b border-border">
        <Sheet open={isSheetOpen} onOpenChange={setIsSheetOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <MoreVertical className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="w-80">
            <SheetHeader>
              <SheetTitle>Conversas</SheetTitle>
            </SheetHeader>
            <div className="mt-4 space-y-2">
              <Button 
                onClick={handleNewChat} 
                className="w-full justify-start gap-2"
                variant="outline"
              >
                <Plus className="h-4 w-4" />
                Novo Chat
              </Button>
              <ScrollArea className="h-[calc(100vh-180px)]">
                <div className="space-y-1 pr-2">
                  {conversations.map((conversation) => (
                    <div
                      key={conversation.id}
                      className={`group flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-colors ${
                        currentConversationId === conversation.id 
                          ? "bg-primary/10 text-primary" 
                          : "hover:bg-accent"
                      }`}
                      onClick={() => editingId !== conversation.id && handleSelectConversation(conversation.id)}
                    >
                      <MessageSquare className="h-4 w-4 flex-shrink-0" />
                      {editingId === conversation.id ? (
                        <input
                          type="text"
                          value={editingTitle}
                          onChange={(e) => setEditingTitle(e.target.value)}
                          onBlur={handleSaveEdit}
                          onKeyDown={handleKeyDown}
                          className="flex-1 bg-background border border-border rounded px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-primary"
                          autoFocus
                          onClick={(e) => e.stopPropagation()}
                        />
                      ) : (
                        <span className="flex-1 text-sm truncate">{conversation.title}</span>
                      )}
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 opacity-0 group-hover:opacity-100"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <MoreVertical className="h-3 w-3" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            onClick={(e) => {
                              e.stopPropagation();
                              handleStartEdit(conversation.id, conversation.title);
                            }}
                          >
                            <Edit3 className="h-4 w-4 mr-2" />
                            Renomear
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            className="text-destructive"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteConversation(conversation.id);
                            }}
                          >
                            <Trash2 className="h-4 w-4 mr-2" />
                            Excluir
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  ))}
                  {conversations.length === 0 && (
                    <div className="text-center py-8 text-muted-foreground text-sm">
                      Nenhuma conversa ainda
                    </div>
                  )}
                </div>
              </ScrollArea>
            </div>
          </SheetContent>
        </Sheet>
      </div>
      
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 flex flex-col h-full">
          {messages.length === 0 ? (
            <div className="flex flex-1 items-center justify-center">
              <span className="text-xl text-muted-foreground font-medium">
                Bem-vindo ao Agente Anti-Fraude!
              </span>
            </div>
          ) : (
            <div className="py-6 space-y-6">
              {messages.map(message => (
                <ChatMessage 
                  key={message.id} 
                  message={message} 
                  isNewMessage={message.id === lastGeneratedMessageId && message.role === "assistant"}
                />
              ))}
              {loading && (
                <div className="flex items-start gap-4">
                  <div className="w-8 h-8 rounded-full overflow-hidden flex items-center justify-center flex-shrink-0">
                    <img 
                      src="/lovable-uploads/iconeIA.jpg" 
                      alt="IA" 
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="typing-indicator mt-1">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              )}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      {/* Chat Input */}
      <div className="border-t border-border bg-card">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <ChatInput 
            onSendMessage={handleSendMessage} 
            onSendAudio={handleSendAudio} 
            isLoading={loading}
            showSuggestions={messages.length === 0}
          />
        </div>
      </div>
    </div>
  );
};

export default Chat;
