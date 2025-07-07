
import { useState, useEffect } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { toast } from 'sonner';

export interface ChatMemoryData {
  user_name: string;
  memories: {
    contexto: {
      loja: string;
      projetos: string[];
      style: string;
    };
    histórico: Array<{
      data: string;
      notes: string;
    }>;
  };
  conversation_title?: string;
  messages?: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
  }>;
}

export interface ChatMemoryRecord {
  id: number;
  user_id: string;
  conversation_id: string;
  loja_id: string;
  created_at: string;
  message: ChatMemoryData;
}

export const useChatMemory = () => {
  const { user, profile } = useAuth();
  const [conversations, setConversations] = useState<ChatMemoryRecord[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [currentMemory, setCurrentMemory] = useState<ChatMemoryData | null>(null);
  const [loading, setLoading] = useState(false);

  // Load conversations when user logs in
  useEffect(() => {
    if (user) {
      loadConversations();
    } else {
      setConversations([]);
      setCurrentConversationId(null);
      setCurrentMemory(null);
    }
  }, [user]);

  // Load current memory when conversation changes
  useEffect(() => {
    if (currentConversationId) {
      loadConversationMemory(currentConversationId);
    } else {
      setCurrentMemory(null);
    }
  }, [currentConversationId]);

  const loadConversations = async () => {
    if (!user) return;

    try {
      const { data, error } = await supabase
        .from('chat_memory')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error loading conversations:', error);
        toast.error('Erro ao carregar conversas');
        return;
      }

      // Group by conversation_id and get the latest record for each conversation
      const conversationMap = new Map<string, ChatMemoryRecord>();
      (data || []).forEach(record => {
        const existing = conversationMap.get(record.conversation_id);
        if (!existing || new Date(record.created_at) > new Date(existing.created_at)) {
          conversationMap.set(record.conversation_id, record as ChatMemoryRecord);
        }
      });

      setConversations(Array.from(conversationMap.values()));
    } catch (error) {
      console.error('Error loading conversations:', error);
      toast.error('Erro ao carregar conversas');
    }
  };

  const loadConversationMemory = async (conversationId: string) => {
    if (!user) return;

    try {
      setLoading(true);
      const { data, error } = await supabase
        .from('chat_memory')
        .select('*')
        .eq('conversation_id', conversationId)
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (error && error.code !== 'PGRST116') {
        console.error('Error loading conversation memory:', error);
        toast.error('Erro ao carregar memória da conversa');
        return;
      }

      if (data) {
        setCurrentMemory(data.message as ChatMemoryData);
      }
    } catch (error) {
      console.error('Error loading conversation memory:', error);
      toast.error('Erro ao carregar memória da conversa');
    } finally {
      setLoading(false);
    }
  };

  const createNewConversation = async (firstMessage?: string) => {
    if (!user) return null;

    try {
      const conversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const title = firstMessage ? 
        firstMessage.split(' ').slice(0, 3).join(' ') + '...' : 
        'Nova conversa';

      const initialMemory: ChatMemoryData = {
        user_name: profile?.full_name || 'Usuário',
        memories: {
          contexto: {
            loja: 'loja 101',
            projetos: ['n8n', 'auditoria lojas', 'SAC E-commerce'],
            style: 'direto, profissional'
          },
          histórico: []
        },
        conversation_title: title,
        messages: []
      };

      const { data, error } = await supabase
        .from('chat_memory')
        .insert({
          user_id: user.id,
          conversation_id: conversationId,
          loja_id: 'loja_101',
          message: initialMemory
        })
        .select()
        .single();

      if (error) {
        console.error('Error creating conversation:', error);
        toast.error('Erro ao criar conversa');
        return null;
      }

      await loadConversations();
      setCurrentConversationId(conversationId);
      return conversationId;
    } catch (error) {
      console.error('Error creating conversation:', error);
      toast.error('Erro ao criar conversa');
      return null;
    }
  };

  const updateConversationMemory = async (conversationId: string, newMessage: { role: 'user' | 'assistant'; content: string }) => {
    if (!user || !currentMemory) return null;

    try {
      const updatedMessages = [
        ...(currentMemory.messages || []),
        {
          role: newMessage.role,
          content: newMessage.content,
          timestamp: new Date().toISOString()
        }
      ];

      const updatedMemory: ChatMemoryData = {
        ...currentMemory,
        messages: updatedMessages
      };

      const { data, error } = await supabase
        .from('chat_memory')
        .insert({
          user_id: user.id,
          conversation_id: conversationId,
          loja_id: 'loja_101',
          message: updatedMemory
        })
        .select()
        .single();

      if (error) {
        console.error('Error updating conversation memory:', error);
        return null;
      }

      setCurrentMemory(updatedMemory);
      return data;
    } catch (error) {
      console.error('Error updating conversation memory:', error);
      return null;
    }
  };

  const deleteConversation = async (conversationId: string) => {
    if (!user) return;

    try {
      const { error } = await supabase
        .from('chat_memory')
        .delete()
        .eq('conversation_id', conversationId)
        .eq('user_id', user.id);

      if (error) {
        console.error('Error deleting conversation:', error);
        toast.error('Erro ao excluir conversa');
        return;
      }

      setConversations(prev => prev.filter(conv => conv.conversation_id !== conversationId));
      
      if (currentConversationId === conversationId) {
        setCurrentConversationId(null);
        setCurrentMemory(null);
      }

      toast.success('Conversa excluída com sucesso');
    } catch (error) {
      console.error('Error deleting conversation:', error);
      toast.error('Erro ao excluir conversa');
    }
  };

  const updateConversationTitle = async (conversationId: string, title: string) => {
    if (!user || !currentMemory) return;

    try {
      const updatedMemory: ChatMemoryData = {
        ...currentMemory,
        conversation_title: title
      };

      const { error } = await supabase
        .from('chat_memory')
        .insert({
          user_id: user.id,
          conversation_id: conversationId,
          loja_id: 'loja_101',
          message: updatedMemory
        });

      if (error) {
        console.error('Error updating conversation title:', error);
        return;
      }

      setCurrentMemory(updatedMemory);
      await loadConversations();
    } catch (error) {
      console.error('Error updating conversation title:', error);
    }
  };

  return {
    conversations,
    currentConversationId,
    currentMemory,
    loading,
    createNewConversation,
    deleteConversation,
    updateConversationMemory,
    updateConversationTitle,
    setCurrentConversationId,
    loadConversations
  };
};
