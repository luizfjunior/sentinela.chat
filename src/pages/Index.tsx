import { useState, useRef, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { useSupabaseConversations } from "@/hooks/useSupabaseConversations";
import ChatMessage from "../components/ChatMessage";
import ChatInput from "../components/ChatInput";
import Sidebar from "../components/Sidebar";
import UserProfile from "../components/UserProfile";
import { toast } from "sonner";
import { useSidebar } from "../hooks/useSidebar";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant";
  timestamp: Date;
  isPending?: boolean; // for messages not yet saved to Supabase
}

const Index = () => {
  const { user, profile } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [lastGeneratedMessageId, setLastGeneratedMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const sidebar = useSidebar();

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
    sidebar.close();
  };

  const handleSelectConversation = (id: string) => {
    setCurrentConversationId(id);
    setMessages([]);
    setLastGeneratedMessageId(null);
    sidebar.close();
  };

  const handleDeleteConversation = (id: string) => {
    deleteConversation(id);
  };

  const handleRenameConversation = async (id: string, newTitle: string) => {
    try {
      await updateConversationTitle(id, newTitle);
      toast.success("Nome da conversa alterado com sucesso");
    } catch (error) {
      console.error("Error renaming conversation:", error);
      toast.error("Erro ao alterar nome da conversa");
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

    // 4. Call backend for AI response with 5-minute timeout
    let assistantContent = "";
    try {
      // Removed timeout to allow unlimited wait time for webhook response
      const response = await fetch("https://n8n.pd.oscarcloud.com.br/webhook-test/ed31b5e4-91f3-", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: content,
          user_id: user.id,
          full_name: profile?.full_name || "Usuário",
          conversation_id: convId
        })
      });
      const responseText = await response.text();

      try {
        if (responseText && responseText.trim()) {
          const data = JSON.parse(responseText);
          assistantContent = data.output || responseText;
        } else {
          assistantContent = "Empty response from server.";
        }
      } catch (parseError) {
        assistantContent = responseText || "Unknown response format.";
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        assistantContent = "A requisição demorou mais de 5 minutos e foi cancelada. Tente novamente.";
        toast.error("Timeout: Requisição cancelada após 5 minutos");
      } else {
        assistantContent = "Erro ao obter resposta da IA.";
        toast.error("Erro ao buscar resposta da IA");
      }
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
          // Removed timeout to allow unlimited wait time for webhook response
          const response = await fetch("https://n8n.pd.oscarcloud.com.br/webhook-test/ed31b5e4-91f3-", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              audioData: base64Audio,
              messageType: "audio",
              user_id: user.id,
              full_name: profile?.full_name || "Usuário",
              conversation_id: convId
            })
          });
          const responseText = await response.text();
          try {
            if (responseText && responseText.trim()) {
              const data = JSON.parse(responseText);
              assistantContent = data.output || responseText;
            } else {
              assistantContent = "Áudio processado com sucesso.";
            }
          } catch {
            assistantContent = responseText || "Áudio recebido e processado.";
          }
        } catch (err: any) {
          if (err.name === 'AbortError') {
            assistantContent = "O processamento do áudio demorou mais de 5 minutos e foi cancelado. Tente novamente.";
            toast.error("Timeout: Processamento de áudio cancelado após 5 minutos");
          } else {
            assistantContent = "Erro ao processar o áudio.";
          }
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
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-[#0f1218] text-gray-900 dark:text-white">
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebar.isOpen}
        onToggle={sidebar.toggle}
        conversations={conversations}
        currentConversationId={currentConversationId}
        onNewChat={handleNewChat}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        onRenameConversation={handleRenameConversation}
      />
      {/* Header */}
      <header className="sticky top-0 z-10 backdrop-blur-md border-b border-gray-200 dark:border-gray-800 bg-zinc-700">
        <div className="max-w-4xl mx-auto flex items-center justify-between px-0 py-[13px]">
          <div className="flex items-center gap-3 ml-12">
            <img 
              alt="Grupo Oscar Logo" 
              src="/lovable-uploads/8d123358-879a-4bd0-8c59-94020f57ed0c.jpg" 
              className="mix-blend-screen w-24 h-auto mx-auto object-fill" 
            />
          </div>
          <UserProfile />
        </div>
      </header>
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto bg-zinc-900">
        <div className="max-w-4xl mx-auto px-4 flex flex-col h-full">
          {messages.length === 0 ? (
            <div className="flex flex-1 items-center justify-center">
              <span className="text-xl text-slate-200 font-medium">
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
      <div className="sticky bottom-0 backdrop-blur-md border-t border-gray-200 dark:border-gray-800 bg-zinc-700">
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

export default Index;
