import discord
from discord.ext import commands
import model

BOT_KEY = 'bot key here'
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
listen_mode = 0 # 0 = none, 1 = respond

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

# react and #inject

@bot.event
async def on_message(message):
    id = message.channel.id
    if id in listening_channels and not message.author.bot and (message.content[0] != '!'):
        if listening_channels[id] != False:

            if(listen_mode == 1):
                emoji = model.Predict(message.content, tokenizer, emojify,unique_emojis, sequence_length)
                await message.channel.send(emoji)

    await bot.process_commands(message)

bot.run(BOT_KEY)
