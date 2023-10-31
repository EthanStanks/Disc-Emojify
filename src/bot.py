import discord
from discord.ext import commands
import model
import re

BOT_KEY = 'discord_bot_key'
padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length = model.DataPrep()
emojify = model.Train(padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length)

bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())

@bot.event
async def on_ready():
    print("Disc-Emoji is now online.")

@bot.command()
async def info(ctx):
    await ctx.send("```- List of Commands -\n!kill - Sets bot to offline.\n!listen - Activates bot in the channel.\n!respond - Bot will respond to each message with emojis\n!stop - Stops bot from listening in the channel.```")

@bot.command()
async def kill(ctx):
    if ctx.author.guild_permissions.administrator:
        await ctx.send("Disc-Emojify is now offline.")
        await bot.close()

listening_channels = {}
listen_mode = 0 # 0 = none, 1 = respond, 2 = inject, 3 = react

@bot.command()
async def listen(ctx):
    if ctx.channel.type == discord.ChannelType.text:
        channel_id = ctx.channel.id
        listening_channels[channel_id] = True
        await ctx.send(f"Now active in this channel.")

@bot.command()
async def stop(ctx):
    if ctx.channel.type == discord.ChannelType.text:
        channel_id = ctx.channel.id
        if channel_id in listening_channels:
            listening_channels[channel_id] = False
            global listen_mode
            listen_mode = 0
            await ctx.send("Now inactive in this channel.")

@bot.command()
async def respond(ctx):
    global listen_mode
    listen_mode = 1
    await ctx.send("Listen mode set: Respond.")

@bot.command()
async def inject(ctx):
    global listen_mode
    listen_mode = 2
    await ctx.send("Listen mode set: Inject.")

@bot.command()
async def react(ctx):
    global listen_mode
    listen_mode = 3
    await ctx.send("Listen mode set: React.")

@bot.event
async def on_message(message):
    id = message.channel.id
    if id in listening_channels and not message.author.bot and (message.content[0] != '!'):
        if listening_channels[id] != False:

            if(listen_mode == 1):
                keyword_emojis = model.Predict(message.content, tokenizer, emojify,unique_emojis, sequence_length)
                respond_emojis = "".join(str(value) for value in keyword_emojis.values())
                if len(keyword_emojis) > 0:
                    await message.channel.send(respond_emojis)
            elif(listen_mode == 2):
                keyword_emojis = model.Predict(message.content, tokenizer, emojify,unique_emojis, sequence_length)
                if len(keyword_emojis) > 0:
                    modified_words = []
                    words = re.findall(r'\b\w+\b|[.,;!?]', message.content)
                    for word in words:
                        if word in keyword_emojis:
                            modified_words.append(keyword_emojis[word])
                        else:
                            modified_words.append(word)
                    injected_message = " ".join(modified_words)
                    await message.channel.send(injected_message)
            elif(listen_mode == 3):
                pass


    await bot.process_commands(message)

bot.run(BOT_KEY)
