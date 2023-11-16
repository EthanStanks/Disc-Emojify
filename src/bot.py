import discord
from discord.ext import commands
import model
import re
import mapping

padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length = model.DataPrep()
emojify,fit = model.Train(padded_sequences, emoji_labels, unique_emojis, tokenizer, sequence_length)

BOT_KEY = 'bot_key'
bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
listening_channels = {}
listen_mode = 0 # 0 = none, 1 = respond, 2 = inject, 3 = react

@bot.event
async def on_ready():
    print("Disc-Emoji is now online.")
    custom_status = discord.Game("Use !info to get started!")
    await bot.change_presence(status=discord.Status.online, activity=custom_status)

@bot.command()
async def info(ctx):
    await ctx.send("List of Commands :placard: :clipboard:\n```!respond - Bot will respond to each message with emojis.\n!replace - Bot will replace keywords in a message with emojis.\n!react - Bot will react to each message with emojis.\n!stop - Stops bot from listening in the channel.\n!kill - Sets bot to offline.```")

@bot.command()
async def kill(ctx):
    if ctx.author.guild_permissions.administrator:
        await ctx.send("Disc-Emojify is now offline :robot: :skull:")
        await bot.close()

@bot.command()
async def stop(ctx):
    if ctx.channel.type == discord.ChannelType.text:
        channel_id = ctx.channel.id
        if channel_id in listening_channels:
            listening_channels[channel_id] = False
            global listen_mode
            listen_mode = 0
            await ctx.send("Now stopping emojis :no_good: :stop_sign:")

@bot.command()
async def respond(ctx):
    if ctx.channel.type == discord.ChannelType.text:
        listening_channels[ctx.channel.id] = True
        global listen_mode
        listen_mode = 1
        await ctx.send("Now responding with emojis :smile: :thought_balloon:")

@bot.command()
async def replace(ctx):
    if ctx.channel.type == discord.ChannelType.text:
        listening_channels[ctx.channel.id] = True
        global listen_mode
        listen_mode = 2
        await ctx.send("Now replacing with emojis :mag: :writing_hand:")

@bot.command()
async def react(ctx):
    if ctx.channel.type == discord.ChannelType.text:
        listening_channels[ctx.channel.id] = True
        global listen_mode
        listen_mode = 3
        await ctx.send("Now reacting with emojis :wave: :grin:")

@bot.event
async def on_message(message):
    id = message.channel.id
    if id in listening_channels and not message.author.bot and (message.content[0] != '!'):
        if listening_channels[id] != False:
            if(listen_mode == 1):
                keyword_emojis, isPredicted = model.Predict(message.content, tokenizer, emojify,unique_emojis, sequence_length)
                if isPredicted:
                    respond_emojis = "".join(str(value) for value in keyword_emojis.values())
                    await message.channel.send(respond_emojis)
                else:
                    await message.add_reaction(mapping.get_emoji((':x:')))
            elif(listen_mode == 2):
                keyword_emojis, isPredicted = model.Predict(message.content, tokenizer, emojify,unique_emojis, sequence_length)
                if isPredicted:
                    modified_words = []
                    words = re.findall(r'\b\w+\b|[.,;!?]', message.content)
                    for word in words:
                        if word in keyword_emojis:
                            modified_words.append(keyword_emojis[word])
                        else:
                            modified_words.append(word)
                    injected_message = " ".join(modified_words)
                    await message.channel.send(injected_message)
                else:
                    await message.add_reaction(mapping.get_emoji((':x:')))
            elif(listen_mode == 3):
                keyword_emojis, isPredicted = model.Predict(message.content, tokenizer, emojify,unique_emojis, sequence_length)
                if isPredicted:
                    for keyword in keyword_emojis:
                        emoji = mapping.get_emoji((keyword_emojis[keyword]))
                        if emoji: await message.add_reaction(emoji)
                        else:
                            print("Could not react with: ",keyword_emojis[keyword],emoji)
                            await message.add_reaction(mapping.get_emoji((':x:')))
                else:
                    await message.add_reaction(mapping.get_emoji((':x:')))            
    await bot.process_commands(message)

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Invalid command :stop_sign: Use **!info** to see the list of available commands :clipboard:")
    elif isinstance(error, commands.MissingPermissions):
        await ctx.send("Unauthorized Permission :face_with_raised_eyebrow: Administrator Only :police_officer:")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("Missing required arguments :stop_sign: Please check the command usage :pencil:")
    else:
        print(f"An error occurred: {error}")

bot.run(BOT_KEY)
