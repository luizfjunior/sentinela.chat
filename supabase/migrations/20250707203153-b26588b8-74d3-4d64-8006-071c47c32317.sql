
-- Criar a nova tabela chat_memory
CREATE TABLE public.chat_memory (
  id SERIAL PRIMARY KEY,
  user_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  loja_id TEXT NOT NULL DEFAULT 'loja_101',
  created_at TIMESTAMP DEFAULT NOW(),
  message JSONB NOT NULL
);

-- Adicionar RLS (Row Level Security)
ALTER TABLE public.chat_memory ENABLE ROW LEVEL SECURITY;

-- Política para permitir usuários verem apenas seus próprios registros
CREATE POLICY "Users can view their own chat memory" 
  ON public.chat_memory 
  FOR SELECT 
  USING (user_id = auth.uid()::text);

-- Política para permitir usuários criarem seus próprios registros
CREATE POLICY "Users can create their own chat memory" 
  ON public.chat_memory 
  FOR INSERT 
  WITH CHECK (user_id = auth.uid()::text);

-- Política para permitir usuários atualizarem seus próprios registros
CREATE POLICY "Users can update their own chat memory" 
  ON public.chat_memory 
  FOR UPDATE 
  USING (user_id = auth.uid()::text);

-- Migrar dados existentes das conversas para a nova estrutura
INSERT INTO public.chat_memory (user_id, conversation_id, loja_id, message, created_at)
SELECT 
  c.user_id::text,
  c.id::text,
  'loja_101' as loja_id,
  jsonb_build_object(
    'user_name', COALESCE(p.full_name, 'Usuário'),
    'memories', jsonb_build_object(
      'contexto', jsonb_build_object(
        'loja', 'loja 101',
        'projetos', jsonb_build_array('n8n', 'auditoria lojas', 'SAC E-commerce'),
        'style', 'direto, profissional'
      ),
      'histórico', jsonb_build_array()
    ),
    'conversation_title', c.title,
    'messages', (
      SELECT jsonb_agg(
        jsonb_build_object(
          'role', m.role,
          'content', m.content,
          'timestamp', m.created_at
        ) ORDER BY m.created_at
      )
      FROM public.messages m 
      WHERE m.conversation_id = c.id
    )
  ) as message,
  c.created_at
FROM public.conversations c
LEFT JOIN public.profiles p ON p.id = c.user_id
WHERE EXISTS (SELECT 1 FROM public.messages m WHERE m.conversation_id = c.id);

-- Criar índices para melhor performance
CREATE INDEX idx_chat_memory_user_id ON public.chat_memory(user_id);
CREATE INDEX idx_chat_memory_conversation_id ON public.chat_memory(conversation_id);
CREATE INDEX idx_chat_memory_loja_id ON public.chat_memory(loja_id);
CREATE INDEX idx_chat_memory_created_at ON public.chat_memory(created_at);
